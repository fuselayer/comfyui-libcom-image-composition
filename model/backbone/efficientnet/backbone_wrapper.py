"""
Wrapper for EfficientNet to use as a backbone that returns multi-scale features
"""

import torch
import torch.nn as nn
from .models import EfficientNet


class EfficientBackboneCommon(nn.Module):
    """
    EfficientNet backbone that returns multi-scale features
    for image harmonization tasks
    """
    
    def __init__(self, model_name='efficientnet-b0', pretrained=False):
        super().__init__()
        
        if pretrained:
            self.model = EfficientNet.from_pretrained(model_name, in_channels=4)
        else:
            self.model = EfficientNet.from_name(model_name, in_channels=4)
    
    @classmethod
    def from_name(cls, model_name, pretrained=False):
        """Create backbone from model name"""
        return cls(model_name=model_name, pretrained=pretrained)
    
    def forward(self, x):
        """
        Forward pass that returns multi-scale features
        
        Args:
            x: Input tensor (B, 4, H, W) - RGB + mask
            
        Returns:
            Tuple of (enc2x, enc4x, enc8x, enc16x, enc32x) features
        """
        endpoints = self.model.extract_endpoints(x)
        
        # Extract features at different scales
        # reduction_1: 2x downsampling
        # reduction_2: 4x downsampling
        # reduction_3: 8x downsampling
        # reduction_4: 16x downsampling
        # reduction_5: 32x downsampling
        
        enc2x = endpoints.get('reduction_1', None)
        enc4x = endpoints.get('reduction_2', None)
        enc8x = endpoints.get('reduction_3', None)
        enc16x = endpoints.get('reduction_4', None)
        enc32x = endpoints.get('reduction_6', None)
        
        return enc2x, enc4x, enc8x, enc16x, enc32x
