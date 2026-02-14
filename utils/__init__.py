# GenMamba-Flow utils

def __getattr__(name):
    if name in ("compute_fid", "compute_psnr_ssim", "compute_bit_accuracy", "compute_lpips"):
        from . import metrics
        return getattr(metrics, name)
    if name in ("save_recovery_grid", "save_stego_comparison", "save_depth_recovery"):
        from . import visualization
        return getattr(visualization, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "compute_fid",
    "compute_psnr_ssim",
    "compute_bit_accuracy",
    "compute_lpips",
    "save_recovery_grid",
    "save_stego_comparison",
    "save_depth_recovery",
]
