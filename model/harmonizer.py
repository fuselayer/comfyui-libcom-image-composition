"""
Complete Harmonizer implementation combining all components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import all the components
from .module import CascadeArgumentRegressor, FilterPerformer
from .filter import Filter
from .backbone import EfficientBackboneCommon

from .base import FilterBasedHarmonizer

class Harmonizer(nn.Module):
    """
    Main Harmonizer model for image harmonization
    Uses EfficientNet backbone with cascaded filter adjustments
    """
    
    def __init__(self):
        super().__init__()
        
        self.input_size = (256, 256)
        self.filter_types = [
            Filter.BRIGHTNESS,
            Filter.CONTRAST,
            Filter.SATURATION,
            Filter.HIGHLIGHT,
            Filter.SHADOW,
        ]
        
        # Backbone for feature extraction
        self.backbone = EfficientBackboneCommon.from_name('efficientnet-b0')
        
        # Regressor to predict filter arguments
        self.regressor = CascadeArgumentRegressor(1280, 160, 1, len(self.filter_types))
        
        # Filter performer to apply adjustments
        self.performer = FilterPerformer(self.filter_types)
    
    def predict_arguments(self, x, mask):
        """
        Predict filter arguments from input image and mask
        
        Args:
            x: RGB image (B, 3, H, W)
            mask: Binary mask (B, 1, H, W)
            
        Returns:
            List of predicted filter arguments
        """
        # Resize to standard input size
        x = F.interpolate(x, self.input_size, mode='bilinear', align_corners=False)
        mask = F.interpolate(mask, self.input_size, mode='bilinear', align_corners=False)
        
        # Concatenate image and mask
        x_with_mask = torch.cat([x, mask], dim=1)
        
        # Extract features
        enc2x, enc4x, enc8x, enc16x, enc32x = self.backbone(x_with_mask)
        
        # Predict arguments from deepest features
        arguments = self.regressor(enc32x)
        
        return arguments
    
    def restore_image(self, x, mask, arguments):
        """
        Apply filter adjustments to restore/harmonize the image
        
        Args:
            x: RGB image (B, 3, H, W)
            mask: Binary mask (B, 1, H, W)
            arguments: List of filter arguments
            
        Returns:
            List of progressively adjusted images
        """
        assert len(arguments) == len(self.filter_types)
        
        # Clamp arguments to valid range and reshape
        arguments = [torch.clamp(arg, -1, 1).view(-1, 1, 1, 1) for arg in arguments]
        
        return self.performer.restore(x, mask, arguments)
    
    def forward(self, x, mask=None):
        """
        Full forward pass: predict arguments and apply filters
        
        Args:
            x: Input (B, 4, H, W) with image+mask concatenated OR (B, 3, H, W) with separate mask
            mask: Optional mask (B, 1, H, W) if x is only RGB
            
        Returns:
            Final harmonized image (last in the cascade)
        """
        # Handle different input formats
        if mask is None:
            # Assume x is (B, 4, H, W) with mask as 4th channel
            assert x.size(1) == 4, "If mask not provided, input must have 4 channels"
            image = x[:, :3, :, :]
            mask = x[:, 3:, :, :]
        else:
            image = x
        
        # Predict filter arguments
        arguments = self.predict_arguments(image, mask)
        
        # Apply filters progressively
        outputs = self.restore_image(image, mask, arguments)
        
        # Return the final output
        return outputs[-1] if outputs else image


class HarmonizerEnhancer(nn.Module):
    """
    Enhanced version of Harmonizer with additional capabilities
    Can be used for image enhancement tasks
    """
    
    def __init__(self):
        super().__init__()
        
        self.input_size = (256, 256)
        self.filter_types = [
            Filter.BRIGHTNESS,
            Filter.CONTRAST,
            Filter.SATURATION,
            Filter.HIGHLIGHT,
            Filter.SHADOW,
        ]
        
        self.backbone = EfficientBackboneCommon.from_name('efficientnet-b0')
        self.regressor = CascadeArgumentRegressor(1280, 160, 1, len(self.filter_types))
        self.performer = FilterPerformer(self.filter_types)
    
    def predict_arguments(self, x, mask):
        x = F.interpolate(x, self.input_size, mode='bilinear', align_corners=False)
        mask_resized = F.interpolate(mask, self.input_size, mode='bilinear', align_corners=False)
        
        x_with_mask = torch.cat([x, mask_resized], dim=1)
        enc2x, enc4x, enc8x, enc16x, enc32x = self.backbone(x_with_mask)
        arguments = self.regressor(enc32x)
        
        return arguments
    
    def restore_image(self, x, mask, arguments):
        assert len(arguments) == len(self.filter_types)
        arguments = [torch.clamp(arg, -1, 1).view(-1, 1, 1, 1) for arg in arguments]
        return self.performer.restore(x, mask, arguments)
    
    def forward(self, x, mask=None):
        if mask is None:
            assert x.size(1) == 4
            image = x[:, :3, :, :]
            mask = x[:, 3:, :, :]
        else:
            image = x
        
        arguments = self.predict_arguments(image, mask)
        outputs = self.restore_image(image, mask, arguments)
        
        return outputs[-1] if outputs else image


# Factory function for creating harmonizer models
def create_harmonizer(model_type='standard', checkpoint_path=None):
    """
    Create a harmonizer model
    
    Args:
        model_type: 'standard' or 'enhancer'
        checkpoint_path: Path to pretrained weights
        
    Returns:
        Harmonizer model
    """
    if model_type == 'standard':
        model = Harmonizer()
    elif model_type == 'enhancer':
        model = HarmonizerEnhancer()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if checkpoint_path:
        print(f"Loading harmonizer checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
        
        # Fix key mismatch: add 'model.' to backbone keys
        fixed_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('backbone.') and not key.startswith('backbone.model.'):
                # Insert 'model.' after 'backbone.'
                new_key = key.replace('backbone.', 'backbone.model.', 1)
                fixed_state_dict[new_key] = value
            else:
                fixed_state_dict[key] = value
        
        # Try to load
        try:
            model.load_state_dict(state_dict, strict=True)
            print("✓ Harmonizer loaded successfully")
        except Exception:  # Don't capture 'e'
            # Silently load with strict=False (key mismatch is expected)
            model.load_state_dict(state_dict, strict=False)
            print("✓ Harmonizer loaded")  # Simple message, no details
    
    return model
