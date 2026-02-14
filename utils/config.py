"""配置加载与合并"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge_config(out[k], v)
        else:
            out[k] = v
    return out


def get_config(config_path: str = "configs/default.yaml", override_path: str = None) -> Dict[str, Any]:
    base = load_config(config_path)
    if override_path and Path(override_path).exists():
        over = load_config(override_path)
        base = merge_config(base, over)
    return base
