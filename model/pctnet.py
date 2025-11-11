# File: custom_nodes/comfyui-libcom-image-composition/model/pctnet.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def _strip_prefix_if_present(state_dict, prefix="module."):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}

def _load_state_dict(core: nn.Module, ckpt_path: str):
    print(f"Loading: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    state = _strip_prefix_if_present(state, "module.")

    # IMPORTANT: load the full state (parameters + buffers). Do NOT filter/mask keys.
    missing, unexpected = core.load_state_dict(state, strict=False)
    if not missing and not unexpected:
        print("PCTNet weights loaded perfectly.")
    else:
        print(f"Loaded with Missing={len(missing)}, Unexpected={len(unexpected)}")
        if missing:
            print("  Missing (sample):", missing[:10])
        if unexpected:
            print("  Unexpected (sample):", unexpected[:10])


# File: custom_nodes/comfyui-libcom-image-composition/model/pctnet.py

# File: custom_nodes/comfyui-libcom-image-composition/model/pctnet.py

class PCTNetAdapter(nn.Module):
    """
    Forward contract: forward(image, mask, strength=1.0, blend_mode='upstream') -> dict
    Process:
      - Reconcile mask to image (dtype/device/shape)
      - Pad both to multiple-of-16 (stability)
      - Downscale to base_size (default 256) for low-res branch
      - Call upstream with low-res+full-res
      - If blend_mode == 'upstream': use upstream images_fullres (or images) as-is
      - Else: recompute full-res transform from params_fullres (or upsampled params), then blend with chosen map (mask/attention)
      - Scale parameters by 'strength' before applying transform
      - Crop final back to original HxW
    """
    def __init__(self, core_model: nn.Module, base_size: int = 256):
        super().__init__()
        self.core = core_model
        self.base_size = base_size

    @torch.no_grad()
    def forward(self, image: torch.Tensor, mask: torch.Tensor | None, strength: float = 1.0, blend_mode: str = "upstream"):
        tried = []

        def split_4ch(x4: torch.Tensor):
            return x4[:, :3, :, :], x4[:, 3:4, :, :]

        def reconcile(rgb: torch.Tensor, m: torch.Tensor):
            if m.ndim == 3: m = m.unsqueeze(1)
            if m.shape[1] != 1: m = m[:, :1, :, :]
            if m.dtype != rgb.dtype: m = m.to(rgb.dtype)
            if m.device != rgb.device: m = m.to(rgb.device)
            if m.shape[2:] != rgb.shape[2:]:
                m = F.interpolate(m, size=rgb.shape[2:], mode="nearest")
            return rgb, m

        def pad_to_multiple(t: torch.Tensor, multiple: int, mode: str = "replicate", value: float = 0.0):
            B, C, H, W = t.shape
            pad_h = (multiple - (H % multiple)) % multiple
            pad_w = (multiple - (W % multiple)) % multiple
            if pad_h == 0 and pad_w == 0:
                return t, (0, 0)
            if mode == "replicate":
                return F.pad(t, (0, pad_w, 0, pad_h), mode="replicate"), (pad_h, pad_w)
            return F.pad(t, (0, pad_w, 0, pad_h), mode="constant", value=value), (pad_h, pad_w)

        def crop_to_original(t: torch.Tensor, pad_hw: tuple[int, int]):
            ph, pw = pad_hw
            if ph == 0 and pw == 0:
                return t
            H, W = t.shape[-2], t.shape[-1]
            return t[..., : H - ph if ph > 0 else H, : W - pw if pw > 0 else W]

        # Normalize input pattern
        if image.ndim == 4 and image.shape[1] == 4 and mask is None:
            rgb, m = split_4ch(image)
        elif image.ndim == 4 and image.shape[1] == 3 and isinstance(mask, torch.Tensor):
            rgb, m = image, mask
        else:
            raise RuntimeError(f"PCTNetAdapter received unsupported shapes image={tuple(image.shape)} mask={type(mask)}")

        # Reconcile and save original size
        rgb, m = reconcile(rgb, m)
        H0, W0 = rgb.shape[2], rgb.shape[3]

        # Pad for stability
        rgb_pad, pad_hw = pad_to_multiple(rgb, multiple=16, mode="replicate")
        m_pad, _ = pad_to_multiple(m, multiple=16, mode="constant", value=0.0)

        # Downscale to base resolution for low-res branch
        rgb_lr = F.interpolate(rgb_pad, size=(self.base_size, self.base_size), mode="bilinear", align_corners=False)
        m_lr = F.interpolate(m_pad, size=(self.base_size, self.base_size), mode="nearest")

        # Call upstream with both low-res and full-res
        try:
            out_dict = self.core(image=rgb_lr, mask=m_lr, image_fullres=rgb_pad, mask_fullres=m_pad)
        except Exception as e:
            tried.append(f"(lowres+fullres kw): {type(e).__name__}: {e}")
            try:
                out_dict = self.core(image=rgb_lr, mask=m_lr)
            except Exception as e2:
                tried.append(f"(lowres only kw): {type(e2).__name__}: {e2}")
                raise RuntimeError("PCTNetAdapter could not call upstream model.\n" + "\n".join(tried))

        # Fetch upstream outputs
        # Normalize dict → tensors
        def to_nchw(val):
            if isinstance(val, torch.Tensor):
                t = val
                if t.ndim == 2: t = t.unsqueeze(0).unsqueeze(0)
                elif t.ndim == 3: t = t.unsqueeze(0)
                return t
            if isinstance(val, (list, tuple)) and val and isinstance(val[0], torch.Tensor):
                elems = []
                for t in val:
                    if t.ndim == 2: t = t.unsqueeze(0).unsqueeze(0)
                    elif t.ndim == 3: t = t.unsqueeze(0)
                    elems.append(t)
                return torch.cat(elems, dim=0)
            return None

        img_upstream = to_nchw(out_dict.get("images_fullres", out_dict.get("images", None)))
        params_lr = to_nchw(out_dict.get("params", None))
        params_fr = to_nchw(out_dict.get("params_fullres", None))
        att_lr = to_nchw(out_dict.get("attention", None))

        # If staying with upstream blend
        if blend_mode == "upstream" and img_upstream is not None:
            tensor_out = img_upstream
        else:
            # Recompute full-res transform and blend by selected map
            # Get/upsample params to full-res pad size
            if params_fr is None and params_lr is not None:
                params_fr = F.interpolate(params_lr, size=rgb_pad.shape[2:], mode="bicubic", align_corners=False)
            if params_fr is None:
                # Fallback to upstream image if params unavailable
                if img_upstream is None:
                    raise RuntimeError("Upstream did not return params(_fullres) or images(_fullres).")
                tensor_out = img_upstream
            else:
                # Scale parameters
                if strength != 1.0:
                    params_fr = params_fr * strength
                # Apply color transform at full-res using the upstream PCT (owned by core)
                transformed = self.core.PCT(rgb_pad, params_fr)
                # Choose blend map
                if blend_mode == "mask":
                    blend_map = m_pad
                elif blend_mode == "attention" and att_lr is not None:
                    blend_map = F.interpolate(att_lr, size=rgb_pad.shape[2:], mode="bilinear", align_corners=False)
                else:
                    # default to upstream image if no proper map; else mask
                    blend_map = m_pad
                tensor_out = transformed * blend_map + rgb_pad * (1.0 - blend_map)

        # Crop back, final resize to original
        tensor_out = crop_to_original(tensor_out, pad_hw)
        if tensor_out.shape[2:] != (H0, W0):
            tensor_out = F.interpolate(tensor_out, size=(H0, W0), mode="bilinear", align_corners=False)
        tensor_out = torch.clamp(tensor_out, 0.0, 1.0)

        return {
            "images": tensor_out,
            "lut_images": tensor_out,
            "base_images": tensor_out,
        }


def _infer_variant_from_path(checkpoint_path: str) -> str:
    name = os.path.basename(checkpoint_path).lower()
    if "vit" in name:
        return "vit"
    if "cnn" in name:
        return "cnn"
    return "cnn"


def _load_state_dict(core: nn.Module, ckpt_path: str):
    print(f"Loading: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    state = _strip_prefix_if_present(state, "module.")
    missing, unexpected = core.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"Loaded with Missing={len(missing)}, Unexpected={len(unexpected)}")
        if unexpected:
            print("  Unexpected (sample):", unexpected[:5])
    else:
        print("PCTNet weights loaded perfectly.")


def create_pctnet(checkpoint_path: str):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"PCTNet checkpoint not found: {checkpoint_path}")

    from .vendor.pctnet_rt import build_pctnet
    variant = _infer_variant_from_path(checkpoint_path)
    core = build_pctnet(variant, checkpoint_path=checkpoint_path)

    # Base sizes used by the released models (adjust if your mconfigs say otherwise)
    base_size = 256  # both CNN and ViT variants in the public repo use 256 during training

    model = PCTNetAdapter(core, base_size=base_size)
    _load_state_dict(core, checkpoint_path)
    return model