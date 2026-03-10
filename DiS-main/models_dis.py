import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
import einops
import sys
import warnings
from functools import partial
from pathlib import Path
from torch import Tensor
from typing import Optional

# Force local vendor Mamba/Causal-Conv1d first to avoid picking site-packages variants.
_PROJECT_ROOT = Path(__file__).resolve().parent
_LOCAL_MAMBA_ROOT = _PROJECT_ROOT / "mamba"
_LOCAL_CAUSAL_ROOT = _PROJECT_ROOT / "causal-conv1d"
for _extra_path in (_LOCAL_MAMBA_ROOT, _LOCAL_CAUSAL_ROOT):
    if _extra_path.exists():
        _p = str(_extra_path)
        if _p not in sys.path:
            sys.path.insert(0, _p)

try:
    from mamba_ssm.modules.mamba_simple import Mamba as _MambaImpl
except Exception:
    _MambaImpl = None

if _MambaImpl is not None:
    try:
        import mamba_ssm.modules.mamba_simple as _mamba_simple_mod
        _mamba_impl_file = Path(_mamba_simple_mod.__file__).resolve()
        if _LOCAL_MAMBA_ROOT not in _mamba_impl_file.parents:
            raise RuntimeError(
                "mamba_ssm is not loaded from local vendor path. "
                f"expected under '{_LOCAL_MAMBA_ROOT}', got '{_mamba_impl_file}'."
            )
        # If core scan op isn't available, the imported Mamba class will fail at runtime.
        if getattr(_mamba_simple_mod, "selective_scan_fn", None) is None:
            raise RuntimeError(
                "Local mamba_ssm loaded but selective_scan_fn is unavailable. "
                "Please reinstall local 'causal-conv1d' and 'mamba'."
            )
    except Exception:
        _MambaImpl = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except Exception:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class _FallbackMamba(nn.Module):
    """Lightweight sequence mixer used when mamba_ssm is unavailable."""

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        **kwargs,
    ):
        super().__init__()
        del d_state, kwargs
        hidden = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, hidden)
        self.dwconv = nn.Conv1d(
            in_channels=hidden,
            out_channels=hidden,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=hidden,
        )
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, d_model)

    def forward(self, x, inference_params=None):
        del inference_params
        h = self.in_proj(x)
        h = self.act(h)
        h = self.dwconv(h.transpose(1, 2)).transpose(1, 2)
        h = self.act(h)
        return self.out_proj(h)


if _MambaImpl is None:
    raise RuntimeError(
        "Failed to load local vendor mamba_ssm from project 'mamba/' folder. "
        "This project is configured to use local Mamba implementation only."
    )
Mamba = _MambaImpl


if RMSNorm is None:
    class RMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-6, device=None, dtype=None):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
            self.bias = None

        def forward(self, x):
            var = x.pow(2).mean(dim=-1, keepdim=True)
            x_norm = x * torch.rsqrt(var + self.eps)
            return x_norm * self.weight


def _fallback_layer_norm_fn(
    x,
    weight,
    bias,
    residual=None,
    prenorm=False,
    residual_in_fp32=False,
    eps=1e-6,
    **kwargs,
):
    del kwargs
    if residual is None:
        residual_out = x
    else:
        residual_out = residual + x
    out = F.layer_norm(
        residual_out.to(dtype=weight.dtype),
        normalized_shape=(weight.shape[0],),
        weight=weight,
        bias=bias,
        eps=eps,
    )
    if prenorm:
        if residual_in_fp32:
            residual_out = residual_out.to(torch.float32)
        return out, residual_out
    return out


def _fallback_rms_norm_fn(
    x,
    weight,
    bias,
    residual=None,
    prenorm=False,
    residual_in_fp32=False,
    eps=1e-6,
    **kwargs,
):
    del kwargs
    if residual is None:
        residual_out = x
    else:
        residual_out = residual + x
    var = residual_out.pow(2).mean(dim=-1, keepdim=True)
    out = residual_out * torch.rsqrt(var + eps) * weight
    if bias is not None:
        out = out + bias
    if prenorm:
        if residual_in_fp32:
            residual_out = residual_out.to(torch.float32)
        return out, residual_out
    return out


if layer_norm_fn is None:
    layer_norm_fn = _fallback_layer_norm_fn
if rms_norm_fn is None:
    rms_norm_fn = _fallback_rms_norm_fn


def haar_dwt2d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x00, x01 = x[:, :, 0::2, 0::2], x[:, :, 0::2, 1::2]
    x10, x11 = x[:, :, 1::2, 0::2], x[:, :, 1::2, 1::2]
    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return ll, lh, hl, hh

def haar_idwt2d(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
    bsz, ch, h, w = ll.shape
    out = ll.new_zeros((bsz, ch, h * 2, w * 2))
    out[:, :, 0::2, 0::2] = (ll + lh + hl + hh) * 0.5
    out[:, :, 0::2, 1::2] = (ll - lh + hl - hh) * 0.5
    out[:, :, 1::2, 0::2] = (ll + lh - hl - hh) * 0.5
    out[:, :, 1::2, 1::2] = (ll - lh - hl + hh) * 0.5
    return out


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding



def patchify(imgs, patch_size):
    x = einops.rearrange(imgs, 'B C (h p1) (w p2) -> B (h w) (p1 p2 C)', p1=patch_size, p2=patch_size)
    return x



def unpatchify(x, channels=3):
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
    x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h, p1=patch_size, p2=patch_size)
    return x


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)



class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)



class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0., skip=False,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, skip=None,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if self.skip_linear is not None:
            hidden_states = self.skip_linear(torch.cat([hidden_states, skip], dim=-1))

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )    
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)



class DualStreamStegoDiSBlock(nn.Module):
    def __init__(
        self, dim, secret_dim=512, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=False,
        bimamba_type="v2", device=None, dtype=None
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        if ssm_cfg is None:
            ssm_cfg = {}

        norm_cls = nn.LayerNorm if not rms_norm else partial(RMSNorm, eps=norm_epsilon)

        self.norm_sem = norm_cls(dim, **factory_kwargs)
        self.norm_tex = norm_cls(dim, **factory_kwargs)

        # 必须显式传入 bimamba_type 保证全局视野
        self.sem_mamba = Mamba(d_model=dim, bimamba_type=bimamba_type, **ssm_cfg, **factory_kwargs)
        self.tex_mamba = Mamba(d_model=dim, bimamba_type=bimamba_type, **ssm_cfg, **factory_kwargs)

        self.adaLN_modulation_sem = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim, bias=True, **factory_kwargs))
        self.adaLN_modulation_tex = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim, bias=True, **factory_kwargs))
        nn.init.constant_(self.adaLN_modulation_sem[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation_sem[-1].bias, 0)
        nn.init.constant_(self.adaLN_modulation_tex[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation_tex[-1].bias, 0)

        self.mask_proj = nn.Linear(dim, dim, **factory_kwargs)
        self.tex_to_z = nn.Linear(dim, dim, **factory_kwargs)

        self.secret_to_x = nn.Linear(secret_dim, dim, **factory_kwargs)
        self.secret_to_z = nn.Linear(secret_dim, dim, **factory_kwargs)
        self.alpha_x = nn.Parameter(torch.tensor(0.01, **factory_kwargs))
        self.alpha_z = nn.Parameter(torch.tensor(0.01, **factory_kwargs))

    def forward(self, h_sem, h_tex, c_global, secret_seq=None, inference_params=None):
        # 1. 双轨 adaLN 参数
        shift_sem, scale_sem, gate_sem = self.adaLN_modulation_sem(c_global).chunk(3, dim=1)
        shift_tex, scale_tex, gate_tex = self.adaLN_modulation_tex(c_global).chunk(3, dim=1)
        shift_sem, scale_sem, gate_sem = shift_sem.unsqueeze(1), scale_sem.unsqueeze(1), gate_sem.unsqueeze(1)
        shift_tex, scale_tex, gate_tex = shift_tex.unsqueeze(1), scale_tex.unsqueeze(1), gate_tex.unsqueeze(1)

        # 2. 语义流 (Master)
        sem_modulated = self.norm_sem(h_sem) * (1 + scale_sem) + shift_sem
        sem_hidden = self.sem_mamba(sem_modulated, inference_params=inference_params)
        h_sem_out = h_sem + gate_sem * sem_hidden

        # 3. 跨流掩码 (Cross-Mask)
        mask = torch.sigmoid(self.mask_proj(h_sem_out))

        # 4. 纹理流 (Slave) 及微扰注入
        tex_modulated = self.norm_tex(h_tex) * (1 + scale_tex) + shift_tex
        x = tex_modulated
        z = self.tex_to_z(tex_modulated)

        if secret_seq is not None:
            x = x + self.alpha_x * torch.tanh(self.secret_to_x(secret_seq))
            z = z + self.alpha_z * torch.tanh(self.secret_to_z(secret_seq))

        tex_hidden = self.tex_mamba(x, inference_params=inference_params)
        z_gate = F.silu(z)

        # 5. 严格的三者相乘门控 (绝对不能用加法)
        tex_merged = tex_hidden * z_gate * mask
        h_tex_out = h_tex + gate_tex * tex_merged

        return h_sem_out, h_tex_out


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    skip=False,
    layer_idx=None,
    device=None,
    dtype=None,
    bimamba_type="none",
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba_type=bimamba_type, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        skip=skip,
    )
    block.layer_idx = layer_idx
    return block



class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, condition=False):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        
        if condition == True: 
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )
        

    def forward(self, x, c=None): 
        if c is not None: 
            c = self.adaLN_modulation(c).squeeze(1)
            shift, scale = c.chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift, scale)
            x = self.linear(x)
        else:
            x = self.norm_final(x)
            x = self.linear(x)
        return x



class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class DisModel(nn.Module): 
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        embed_dim=192,
        channels=3,
        depth=12,
        num_classes=-1,
        ssm_cfg=None,
        drop_rate=0.,
        drop_path_rate=0.1,
        norm_epsilon: float = 1e-5, 
        rms_norm: bool = True, 
        initializer_cfg=None,
        fused_add_norm=True,
        residual_in_fp32=True,
        device=None,
        dtype=None,
        bimamba_type="v2",
        learn_sigma=True,
        if_cls_token=False,
        mlp_time_embed=True,
        conv=True, skip=True,
        class_dropout_prob=0.1,
        **kwargs
    ): 
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.channels = channels
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs) 

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=channels, embed_dim=embed_dim)
        num_patches = (img_size // patch_size) ** 2
        
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()
        
        if self.num_classes > 0:
            # self.label_emb = nn.Embedding(self.num_classes, embed_dim)
            self.label_emb = LabelEmbedder(num_classes, embed_dim, class_dropout_prob)
            self.extras = 2
        else:
            self.extras = 1

        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, embed_dim)) 

         # TODO: release this comment
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
        self.in_blocks = nn.ModuleList([
            create_block(
                embed_dim,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
                bimamba_type=bimamba_type,
                drop_path=inter_dpr[i],
                **factory_kwargs,
            )
            for i in range(depth // 2)])

        self.mid_block = create_block(
                embed_dim,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=depth // 2,
                bimamba_type=bimamba_type,
                drop_path=inter_dpr[depth // 2],
                **factory_kwargs,
            )

        self.out_blocks = nn.ModuleList([
            create_block(
                embed_dim,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i + depth // 2 + 1,
                bimamba_type=bimamba_type,
                drop_path=inter_dpr[i + depth // 2 + 1],
                skip=skip,
                **factory_kwargs,
            )
            for i in range(depth // 2)])

        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        if learn_sigma == True: 
            self.patch_dim = patch_size ** 2 * channels * 2
        else: 
            self.patch_dim = patch_size ** 2 * channels
        
        self.out_channels = channels * 2 if learn_sigma else channels
        # self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias=True) 

        if num_classes > 0:
            self.final_layer = FinalLayer(embed_dim, patch_size, self.out_channels, condition=True)
        else:
            self.final_layer = FinalLayer(embed_dim, patch_size, self.out_channels)    
        # original init
        # self.apply(segm_init_weights) 
        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}
    
    def forward(self, x, timesteps, labels=None, inference_params=None): 
        x = self.patch_embed(x)
        B, L, D = x.shape

        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1)
        
        x = torch.cat((time_token, x), dim=1)

        if labels is not None:
            label_emb = self.label_emb(labels, self.training)
            label_emb = label_emb.unsqueeze(dim=1)
            x = torch.cat((label_emb, x), dim=1) 

        x = x + self.pos_embed

        # mamba impl
        residual = None
        hidden_states = x

        skips = []
        for blk in self.in_blocks: 
            hidden_states, residual = blk(hidden_states, residual, inference_params=inference_params) 
            skips.append(hidden_states)

        hidden_states, residual = self.mid_block(hidden_states, residual, inference_params=inference_params)

        for blk in self.out_blocks:
            hidden_states, residual = blk(hidden_states, residual, inference_params=inference_params, skip=skips.pop())

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        
        x = hidden_states
        # x = self.norm_f(hidden_states)
        # x = self.decoder_pred(x)
        if labels is not None:
            x = self.final_layer(x, c=time_token+label_emb)
        else:
            x = self.final_layer(x)

        assert x.size(1) == self.extras + L
        x = x[:, self.extras:, :]
        x = unpatchify(x, self.out_channels)
        # x = self.final_layer(x)
        return x


    def forward_with_cfg(self, x, timesteps, cfg_scale=1.5, labels=None):

        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, timesteps, labels)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)



class StegoDiSModel(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        embed_dim=192,
        channels=3,
        depth=12,
        num_classes=-1,
        secret_dim=512,
        ssm_cfg=None,
        drop_rate=0.,
        drop_path_rate=0.1,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        initializer_cfg=None,
        fused_add_norm=True,
        residual_in_fp32=True,
        device=None,
        dtype=None,
        bimamba_type="v2",
        learn_sigma=True,
        if_cls_token=False,
        mlp_time_embed=True,
        conv=True, skip=True,
        class_dropout_prob=0.1,
        **kwargs
    ):
        super().__init__()
        del drop_rate, drop_path_rate, if_cls_token, conv, skip
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.channels = channels
        self.depth = depth
        self.secret_dim = secret_dim
        self.patch_size = patch_size
        self.lowres_patch_size = max(1, patch_size // 2)
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs)

        if ssm_cfg is None:
            ssm_cfg = {}

        lowres_img_size = img_size // 2
        num_patches = (lowres_img_size // self.lowres_patch_size) ** 2
        norm_cls = partial(
            nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
        )

        self.sem_in_proj = PatchEmbed(
            patch_size=self.lowres_patch_size,
            in_chans=channels,
            embed_dim=embed_dim,
        )
        self.tex_in_proj = PatchEmbed(
            patch_size=self.lowres_patch_size,
            in_chans=channels * 3,
            embed_dim=embed_dim,
        )

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.blocks = nn.ModuleList([
            DualStreamStegoDiSBlock(
                embed_dim,
                secret_dim=secret_dim,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                bimamba_type=bimamba_type,
                **factory_kwargs,
            )
            for _ in range(depth)
        ])

        self.skip_sem_proj = nn.ModuleList([
            nn.Linear(2 * embed_dim, embed_dim, **factory_kwargs)
            for _ in range(depth // 2)
        ])
        self.skip_tex_proj = nn.ModuleList([
            nn.Linear(2 * embed_dim, embed_dim, **factory_kwargs)
            for _ in range(depth // 2)
        ])

        self.norm_sem_f = norm_cls(embed_dim)
        self.norm_tex_f = norm_cls(embed_dim)

        self.sem_out_proj = nn.Linear(
            embed_dim,
            self.lowres_patch_size ** 2 * embed_dim,
            **factory_kwargs,
        )
        self.tex_out_proj = nn.Linear(
            embed_dim,
            self.lowres_patch_size ** 2 * embed_dim * 3,
            **factory_kwargs,
        )

        self.final_patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
        )

        self.out_channels = channels * 2 if learn_sigma else channels
        if self.num_classes > 0:
            self.label_emb = LabelEmbedder(num_classes, embed_dim, class_dropout_prob)
            self.final_layer = FinalLayer(embed_dim, patch_size, self.out_channels, condition=True)
        else:
            self.final_layer = FinalLayer(embed_dim, patch_size, self.out_channels)

        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def _prepare_secret_seq(self, secret_seq, seq_len):
        if secret_seq is None:
            return None
        if secret_seq.dim() == 2:
            if secret_seq.shape[-1] != self.secret_dim:
                raise ValueError(f"Expected secret_seq last dim {self.secret_dim}, got {secret_seq.shape[-1]}")
            return secret_seq.unsqueeze(1).expand(-1, seq_len, -1)
        if secret_seq.dim() != 3:
            raise ValueError("secret_seq must have shape [B, secret_dim] or [B, L, secret_dim].")
        if secret_seq.shape[-1] != self.secret_dim:
            raise ValueError(f"Expected secret_seq last dim {self.secret_dim}, got {secret_seq.shape[-1]}")
        if secret_seq.shape[1] == 1:
            return secret_seq.expand(-1, seq_len, -1)
        if secret_seq.shape[1] != seq_len:
            secret_seq = F.interpolate(
                secret_seq.transpose(1, 2),
                size=seq_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return secret_seq.contiguous()

    def forward(self, x, timesteps=None, labels=None, secret_seq=None, inference_params=None, t=None):
        if timesteps is None:
            timesteps = t
        if timesteps is None:
            raise ValueError("StegoDiSModel.forward requires `timesteps` or alias `t`.")
        ll, lh, hl, hh = haar_dwt2d(x)
        tex = torch.cat([lh, hl, hh], dim=1)

        h_sem = self.sem_in_proj(ll)
        h_tex = self.tex_in_proj(tex)
        B, L, D = h_sem.shape
        assert h_tex.shape == (B, L, D)

        c_global = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        if labels is not None:
            label_emb = self.label_emb(labels, self.training)
            c_global = c_global + label_emb

        h_sem = h_sem + self.pos_embed
        h_tex = h_tex + self.pos_embed
        secret_seq = self._prepare_secret_seq(secret_seq, L)

        skip_sem = []
        skip_tex = []
        for i, blk in enumerate(self.blocks):
            if i >= self.depth // 2:
                h_sem = self.skip_sem_proj[i - self.depth // 2](torch.cat([skip_sem.pop(), h_sem], dim=-1))
                h_tex = self.skip_tex_proj[i - self.depth // 2](torch.cat([skip_tex.pop(), h_tex], dim=-1))

            h_sem, h_tex = blk(
                h_sem,
                h_tex,
                c_global,
                secret_seq=secret_seq,
                inference_params=inference_params,
            )

            if i < self.depth // 2:
                skip_sem.append(h_sem)
                skip_tex.append(h_tex)

        h_sem = self.norm_sem_f(h_sem)
        h_tex = self.norm_tex_f(h_tex)

        ll_feat = unpatchify(self.sem_out_proj(h_sem), self.embed_dim)
        tex_feat = unpatchify(self.tex_out_proj(h_tex), self.embed_dim * 3)
        lh_feat, hl_feat, hh_feat = tex_feat.chunk(3, dim=1)

        fused_feat = haar_idwt2d(ll_feat, lh_feat, hl_feat, hh_feat)
        fused_tokens = self.final_patch_embed(fused_feat)

        if self.num_classes > 0:
            v_pred = self.final_layer(fused_tokens, c=c_global)
        else:
            v_pred = self.final_layer(fused_tokens)

        v_pred = unpatchify(v_pred, self.out_channels)
        return v_pred

    def forward_with_cfg(self, x, timesteps, cfg_scale=1.5, labels=None, secret_seq=None):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)

        if secret_seq is not None:
            secret_half = secret_seq[: len(secret_seq) // 2]
            secret_seq = torch.cat([secret_half, secret_half], dim=0)

        model_out = self.forward(combined, timesteps, labels, secret_seq=secret_seq)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


def dis_s_2(**kwargs): 
    model = DisModel(
        patch_size=2,
        embed_dim=368,
        depth=24,
        **kwargs
    )
    return model 


def dis_s_4(**kwargs): 
    model = DisModel(
        patch_size=4,
        embed_dim=368,
        depth=24,
        **kwargs
    )
    return model 


def dis_b_2(**kwargs): 
    model = DisModel(
        patch_size=2,
        embed_dim=768,
        depth=24,
        **kwargs
    )
    return model 


def dis_b_4(**kwargs): 
    model = DisModel(
        patch_size=4,
        embed_dim=768,
        depth=24,
        **kwargs
    )
    return model 


def dis_l_2(**kwargs): 
    model = DisModel(
        patch_size=2,
        embed_dim=1024,
        depth=48,
        **kwargs
    )
    return model 


def dis_l_4(**kwargs): 
    model = DisModel(
        patch_size=4,
        embed_dim=1024,
        depth=48,
        **kwargs
    )
    return model 


def dis_m_2(**kwargs): 
    model = DisModel(
        patch_size=2,
        embed_dim=768,
        depth=48,
        **kwargs
    )
    return model 

def dis_m_4(**kwargs): 
    model = DisModel(
        patch_size=4,
        embed_dim=768,
        depth=48,
        **kwargs
    )
    return model 


def dis_h_2(**kwargs): 
    model = DisModel(
        patch_size=2,
        embed_dim=1536,
        depth=48,
        **kwargs
    )
    return model 

def dis_h_4(**kwargs): 
    model = DisModel(
        patch_size=4,
        embed_dim=1536,
        depth=48,
        **kwargs
    )
    return model 



DiS_models = {
    "DiS-S/2": dis_s_2, "DiS-S/4": dis_s_4,
    "DiS-B/2": dis_b_2, "DiS-B/4": dis_b_4,
    "DiS-M/2": dis_m_2, "DiS-M/4": dis_m_4, 
    "DiS-L/2": dis_l_2, "DiS-L/4": dis_l_4,
    "DiS-H/2": dis_h_2, "DiS-H/4": dis_h_4, 
}