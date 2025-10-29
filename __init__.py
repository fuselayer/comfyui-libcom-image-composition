"""
ComfyUI Image Harmonization Custom Node
Multi-model support for CDTNet and Harmonizer
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Version info
__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Multi-model image harmonization for ComfyUI"

print(f"[Image Harmonization] Loaded v{__version__}")
print("[Image Harmonization] Available models:")
print("  - CDTNet_iHarmony4_256 (256x256, with refinement)")
print("  - CDTNet_HAdobe5k_2048 (2048x2048, with refinement)")
print("  - CDTNet_sim_base256 (256x256, simplified/fast)")
print("  - Harmonizer_iHarmony4 (256x256, EfficientNet-based)")