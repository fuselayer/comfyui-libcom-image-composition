import os
import sys
from typing import Literal, Optional
import torch
import torch.nn as nn

# Ensure vendored 'upstream' is on sys.path before importing iharm.*
UPSTREAM_ROOT = os.path.join(os.path.dirname(__file__), "upstream")
if UPSTREAM_ROOT not in sys.path:
    sys.path.insert(0, UPSTREAM_ROOT)


def build_pctnet(variant: Literal["cnn", "vit"], checkpoint_path: Optional[str] = None):
    """
    Build PCTNet using the exact training config from BMCONFIGS['...']['params'].
    No guessing: we pass cfg['params'] verbatim into PCTNet(**params).
    """
    from iharm.mconfigs import BMCONFIGS
    from iharm.model.base.pct_net import PCTNet

    # Pick a key by variant. Prefer canonical names, fallback to a backbone-type scan.
    key = None
    if variant == "cnn":
        if "CNN_pct" in BMCONFIGS:
            key = "CNN_pct"
        else:
            # fallback: first config whose params.backbone_type == 'ssam'
            for k, v in BMCONFIGS.items():
                params = getattr(v, "get", lambda _: None)("params") if isinstance(v, dict) else getattr(v, "params", None)
                if isinstance(params, dict) and params.get("backbone_type", "").lower() == "ssam":
                    key = k
                    break
    else:
        if "ViT_pct" in BMCONFIGS:
            key = "ViT_pct"
        else:
            # fallback: first config whose params.backbone_type == 'vit'
            for k, v in BMCONFIGS.items():
                params = getattr(v, "get", lambda _: None)("params") if isinstance(v, dict) else getattr(v, "params", None)
                if isinstance(params, dict) and params.get("backbone_type", "").lower().startswith("vit"):
                    key = k
                    break

    if key is None:
        raise RuntimeError(f"[PCTNet vendor] Could not find a BMCONFIGS entry matching variant='{variant}'")

    cfg = BMCONFIGS[key]
    params = cfg["params"] if isinstance(cfg, dict) else getattr(cfg, "params", None)
    if not isinstance(params, dict):
        raise RuntimeError(f"[PCTNet vendor] BMCONFIGS['{key}'] has no 'params' dict")

    # Instantiate exactly as trained
    model = PCTNet(**params)

    # One-time visibility (remove if noisy)
    print("[PCTNet vendor kwargs]", {
        "variant": variant,
        **{k: params.get(k) for k in ("backbone_type", "transform_type", "color_space", "use_attn", "image_fusion", "ch", "depth")}
    })

    return model