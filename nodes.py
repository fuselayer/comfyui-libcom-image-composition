"""
ComfyUI Custom Node for Multi-Model Image Harmonization
Supports CDTNet and Harmonizer models
"""
from .utils.model_registry import ModelRegistry
import torch
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image

import folder_paths
import comfy.model_management as mm



class HarmonyScoreNode:
    """
    Evaluate harmony score of a composite image
    Uses BargainNet to measure style consistency between foreground/background
    """
    
    def __init__(self):
        self.scorer = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "composite_image": ("IMAGE",),
                "mask": ("MASK",),
            },
            "optional": {
                "harmonized_image": ("IMAGE",),  # Optional: compare before/after
            }
        }
    
    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("harmony_score", "score_text")
    FUNCTION = "evaluate_harmony"
    CATEGORY = "image/harmonization"
    
    def load_scorer(self):
        """Lazy load the harmony scorer"""
        if self.scorer is None:
            from .model.evaluation import HarmonyScorer
            self.scorer = HarmonyScorer(device=mm.get_torch_device())
        return self.scorer
    
    def evaluate_harmony(self, composite_image, mask, harmonized_image=None):
        """
        Evaluate harmony score
        
        Args:
            composite_image: Original composite
            mask: Object mask
            harmonized_image: Optional harmonized result (for comparison)
        
        Returns:
            harmony_score: Float 0-1 (higher = better)
            score_text: Formatted text description
        """
        scorer = self.load_scorer()
        
        # Score the composite
        composite_score = scorer.score(composite_image, mask)
        
        if harmonized_image is not None:
            # Score the harmonized version
            harmonized_score = scorer.score(harmonized_image, mask)
            
            # Calculate improvement
            improvement = harmonized_score - composite_score
            
            # Format output
            score_text = (
                f"Composite Score: {composite_score:.4f}\n"
                f"Harmonized Score: {harmonized_score:.4f}\n"
                f"Improvement: {improvement:+.4f}"
            )
            
            print(score_text)
            return (harmonized_score, score_text)
        
        else:
            # Only composite score
            score_text = f"Harmony Score: {composite_score:.4f}"
            print(score_text)
            return (composite_score, score_text)


class ImageHarmonizationNode:
    """
    ComfyUI node for image harmonization using multiple models
    """
    
    
    def __init__(self):
        self.model = None
        self.current_model_name = None
        self.device = mm.get_torch_device()
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "composite_image": ("IMAGE",),
                "mask": ("MASK",),
                "model_variant": (ModelRegistry.get_all_models(),),
            },
            "optional": {
                # Always visible; ignored for non-PCTNet models
                "pctnet_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "pctnet_blend_mode": (["upstream", "mask", "attention"], {"default": "upstream"}),
                # Keep max_resolution visible here too
                "max_resolution": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 256}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("harmonized_image",)
    FUNCTION = "harmonize"
    CATEGORY = "image/harmonization"
    
    def load_model(self, model_name):
        # Get model config from registry
        config = ModelRegistry.get_config(model_name)
        model_type = config.type

        # Checkpoints live ONLY in this node's ./checkpoints
        node_root = os.path.dirname(os.path.abspath(__file__))
        checkpoints_dir = os.path.join(node_root, "checkpoints")
        checkpoint_path = os.path.join(checkpoints_dir, config.checkpoint)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Expected all harmonization weights in: {checkpoints_dir}"
            )

        # Load model based on type
        if model_type == "cdtnet":
            from .model.cdtnet import create_cdtnet
            self.model = create_cdtnet(
                model_name=model_name,
                checkpoint_path=checkpoint_path
            )

        elif model_type == "issam":
            from .model.issam import create_issam
            self.model = create_issam(checkpoint_path=checkpoint_path)

        elif model_type == "pctnet":
            from .model.pctnet import create_pctnet
            self.model = create_pctnet(checkpoint_path=checkpoint_path)

        elif model_type == "harmonizer":
            from .model.harmonizer import create_harmonizer
            self.model = create_harmonizer(model_type='standard', checkpoint_path=checkpoint_path)

        elif model_type == "enhancer":
            from .model.harmonizer import create_harmonizer
            self.model = create_harmonizer(model_type='enhancer', checkpoint_path=checkpoint_path)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model = self.model.to(self.device)
        self.model.eval()
        self.current_model_name = model_name
        return self.model
    
    def preprocess_inputs(self, composite_image, mask):
        """
        Convert ComfyUI tensors to model input format
        
        Args:
            composite_image: ComfyUI IMAGE tensor (B, H, W, C) in [0, 1]
            mask: ComfyUI MASK tensor (B, H, W) or (H, W) in [0, 1]
        
        Returns:
            comp_img: torch.Tensor (B, 3, H, W) in [0, 1]
            mask_tensor: torch.Tensor (B, 1, H, W) in [0, 1]
        """
        # Handle batch dimension for composite image
        if composite_image.dim() == 3:
            composite_image = composite_image.unsqueeze(0)
        
        # Convert from (B, H, W, C) to (B, C, H, W)
        comp_img = composite_image.permute(0, 3, 1, 2).to(self.device)
        
        # Handle mask
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        
        mask_tensor = mask.to(self.device)
        
        # Ensure mask matches image batch size
        if mask_tensor.size(0) == 1 and comp_img.size(0) > 1:
            mask_tensor = mask_tensor.expand(comp_img.size(0), -1, -1, -1)
        
        # Resize mask to match image if needed
        if mask_tensor.shape[2:] != comp_img.shape[2:]:
            mask_tensor = F.interpolate(
                mask_tensor,
                size=comp_img.shape[2:],
                mode='nearest'
            )
        
        return comp_img, mask_tensor
    
    def postprocess_output(self, output):
        """
        Convert model output to ComfyUI format
        
        Args:
            output: torch.Tensor (B, 3, H, W) in [0, 1]
        
        Returns:
            ComfyUI IMAGE tensor (B, H, W, C) in [0, 1]
        """
        # Convert from (B, C, H, W) to (B, H, W, C)
        output = output.permute(0, 2, 3, 1)
        
        # Ensure values are in [0, 1]
        output = torch.clamp(output, 0, 1)
        
        return output.cpu()
    
    def resize_for_processing(self, comp_img, mask_tensor, max_resolution):
        """
        Resize inputs if they exceed max_resolution while maintaining aspect ratio
        
        Returns:
            resized images, original size for upsampling later
        """
        original_size = comp_img.shape[2:]  # (H, W)
        
        # Check if resizing is needed
        max_dim = max(original_size)
        if max_dim <= max_resolution:
            return comp_img, mask_tensor, original_size
        
        # Calculate new size maintaining aspect ratio
        scale = max_resolution / max_dim
        new_h = int(original_size[0] * scale)
        new_w = int(original_size[1] * scale)
        
        # Make dimensions divisible by 8 for better processing
        new_h = (new_h // 8) * 8
        new_w = (new_w // 8) * 8
        
        # Resize
        comp_img_resized = F.interpolate(
            comp_img,
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        )
        
        mask_resized = F.interpolate(
            mask_tensor,
            size=(new_h, new_w),
            mode='nearest'
        )
        
        return comp_img_resized, mask_resized, original_size
    
    def harmonize(self, composite_image, mask, model_variant, max_resolution=1024, pctnet_strength=1.0, pctnet_blend_mode="upstream"):
        """
        Main harmonization function
        
        Args:
            composite_image: ComfyUI IMAGE tensor
            mask: ComfyUI MASK tensor
            model_variant: Name of the model to use
            max_resolution: Maximum resolution for processing
        
        Returns:
            Harmonized image in ComfyUI format
        """
        # Load model
        model = self.load_model(model_variant)
        config = ModelRegistry.get_config(model_variant)
        
        # Preprocess inputs
        comp_img, mask_tensor = self.preprocess_inputs(composite_image, mask)
        
        # Resize if necessary
        comp_img_proc, mask_proc, original_size = self.resize_for_processing(
            comp_img, mask_tensor, max_resolution
        )
        
        
        # Run inference
        with torch.no_grad():
            if config.type in ["cdtnet", "issam"]:
                # CDTNet forward pass - returns dict with 'images', 'lut_images', 'base_images'
                result = model(comp_img_proc, mask_proc)
                base_output = result['base_images']      # ADD THIS
                lut_output = result['lut_images']        # ADD THIS
                output = result['images']                 # Use the final harmonized output
                
                print(f"Base output range: [{base_output.min():.3f}, {base_output.max():.3f}]")
                print(f"LUT output range: [{lut_output.min():.3f}, {lut_output.max():.3f}]")
                print(f"Final output range: [{output.min():.3f}, {output.max():.3f}]")
                
            elif config.type == "harmonizer":
                # Harmonizer forward pass
                model_input = torch.cat([comp_img_proc, mask_proc], dim=1)
                output = model(model_input)

            elif config.type == "pctnet":
                result = model(comp_img_proc, mask_proc,
                               strength=pctnet_strength,
                               blend_mode=pctnet_blend_mode)
                output = result['images'] if isinstance(result, dict) and 'images' in result else result
                
            elif config.type == "enhancer":  # CHANGED: was checking "harmonizer" again
                # Enhancer forward pass
                model_input = torch.cat([comp_img_proc, mask_proc], dim=1)
                output = model(model_input)
            
            
# Save all three for comparison        
        
        # Resize back to original size if needed
        if output.shape[2:] != original_size:
            output = F.interpolate(
                output,
                size=original_size,
                mode='bilinear',
                align_corners=False
            )
        
        # Postprocess to ComfyUI format
        result = self.postprocess_output(output)
        
        return (result,)


# Optional: Add a simpler node that auto-detects best model
class ImageHarmonizationAuto:
    """
    Simplified harmonization node that automatically selects the best model
    based on input resolution
    """
    
    def __init__(self):
        self.base_node = ImageHarmonizationNode()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "composite_image": ("IMAGE",),
                "mask": ("MASK",),
            },
            "optional": {
                "prefer_quality": ("BOOLEAN", {"default": True}),
                "max_resolution": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 256}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("harmonized_image",)
    FUNCTION = "harmonize_auto"
    CATEGORY = "image/harmonization"
    
    def harmonize_auto(self, composite_image, mask, prefer_quality=True, max_resolution=1024):
        """
        Automatically select and run the best model
        """
        # Determine input resolution
        if composite_image.dim() == 3:
            h, w = composite_image.shape[0], composite_image.shape[1]
        else:
            h, w = composite_image.shape[1], composite_image.shape[2]
        
        max_dim = max(h, w)
        
        # Select model based on resolution and quality preference
        if max_dim > 1024 and prefer_quality:
            model_variant = "CDTNet_HAdobe5k_2048"
        elif prefer_quality:
            model_variant = "CDTNet_iHarmony4_256"
        else:
            model_variant = "CDTNet_sim_base256"  # Faster, simplified model
        
        print(f"Auto-selected model: {model_variant} for {max_dim}px input")
        
        # Use base node to process
        return self.base_node.harmonize(
            composite_image, 
            mask, 
            model_variant, 
            max_resolution
        )


NODE_CLASS_MAPPINGS = {
    "ImageHarmonization": ImageHarmonizationNode,
    "ImageHarmonizationAuto": ImageHarmonizationAuto,
    "HarmonyScore": HarmonyScoreNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageHarmonization": "Image Harmonization (Multi-Model)",
    "ImageHarmonizationAuto": "Image Harmonization (Auto)",
    "HarmonyScore": "Harmony Score Evaluation"
}


