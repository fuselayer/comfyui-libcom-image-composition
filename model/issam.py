"""
iSSAM - Image-Specific Spatial Adaptation Module
Simpler and faster variant of CDTNet without refinement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from .base import CDTNetBase


# ============================================================================
# Reuse components from CDTNet
# ============================================================================

# Import from cdtnet.py (make sure these are importable)
from .cdtnet import (
    UNetEncoder,
    UNetDecoder,
    SpatialSeparatedAttention,
    Generator3DLUT_identity,
    Generator3DLUT_zero,
    TrilinearInterpolation,
)


# ============================================================================
# iSSAM-specific LUT Weight Predictor
# ============================================================================

class Weight_predictor_iSSAM(nn.Module):
    """
    Simpler weight predictor for iSSAM
    Only uses deepest encoder features (not multi-scale)
    """
    
    def __init__(self, in_channels=256, out_channels=3, fb=True):
        super().__init__()
        self.mid_ch = 256
        self.fb = fb  # foreground-background separation
        
        self.conv = nn.Conv2d(in_channels, self.mid_ch, 1, 1, padding=0)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        
        if self.fb:
            # Separate features for foreground and background
            self.fc = nn.Conv2d(self.mid_ch * 2, out_channels, 1, 1, padding=0)
        else:
            self.fc = nn.Conv2d(self.mid_ch, out_channels, 1, 1, padding=0)
    
    def forward(self, encoder_outputs, mask):
        """
        Predict LUT weights from encoder features
        
        Args:
            encoder_outputs: List of encoder features (we use only [0])
            mask: Binary mask (B, 1, H, W)
        
        Returns:
            weights: LUT mixing weights (B, n_lut)
        """
        fea_input = encoder_outputs[0]  # Only use deepest features
        x = self.conv(fea_input)
        
        if self.fb:
            # Separate foreground/background features
            down_mask = F.interpolate(mask, size=fea_input.shape[2:], mode='bilinear')
            fg_feature = self.avg_pooling(x * down_mask)
            bg_feature = self.avg_pooling(x * (1 - down_mask))
            fgbg_fea = torch.cat((fg_feature, bg_feature), 1)
            x = self.fc(fgbg_fea)
        else:
            feature = self.avg_pooling(x)
            x = self.fc(feature)
        
        return x


class LUT_iSSAM(nn.Module):
    """
    LUT module for iSSAM
    Uses simpler weight predictor
    """
    
    def __init__(self, in_channels=256, n_lut=4, fb=True):
        super().__init__()
        self.n_lut = n_lut
        self.fb = fb
        
        # Initialize LUTs
        if n_lut >= 1:
            self.LUT0 = Generator3DLUT_identity()
        if n_lut >= 2:
            self.LUT1 = Generator3DLUT_zero()
        if n_lut >= 3:
            self.LUT2 = Generator3DLUT_zero()
        if n_lut >= 4:
            self.LUT3 = Generator3DLUT_zero()
        
        self.classifier = Weight_predictor_iSSAM(in_channels, n_lut, fb)
    
    def forward(self, encoder_outputs, image, mask):
        """
        Apply learned LUT transformation
        
        Returns:
            Harmonized image with LUT applied to masked region
        """
        pred_weights = self.classifier(encoder_outputs, mask)
        
        if len(pred_weights.shape) == 1:
            pred_weights = pred_weights.unsqueeze(0)
        
        combine_A = image.new(image.size())
        
        if self.n_lut == 4:
            gen_A0 = self.LUT0(image)
            gen_A1 = self.LUT1(image)
            gen_A2 = self.LUT2(image)
            gen_A3 = self.LUT3(image)
            
            for b in range(image.size(0)):
                combine_A[b, :, :, :] = (
                    pred_weights[b, 0] * gen_A0[b, :, :, :] +
                    pred_weights[b, 1] * gen_A1[b, :, :, :] +
                    pred_weights[b, 2] * gen_A2[b, :, :, :] +
                    pred_weights[b, 3] * gen_A3[b, :, :, :]
                )
        
        # Apply only to masked region
        combine_A = combine_A * mask + image * (1 - mask)
        
        return combine_A


# ============================================================================
# iSSAM Model
# ============================================================================

class iSSAM(CDTNetBase):
    """
    iSSAM: Simplified harmonization model
    - Uses encoder/decoder only
    - NO LUT (uses decoder output directly)
    - NO refinement module (faster)
    """
    
    def __init__(self, depth=4, ch=32, max_channels=512,
                 norm_layer=nn.BatchNorm2d, batchnorm_from=2,
                 attend_from=3, attention_mid_k=2.0,
                 image_fusion=True):  # Removed n_lut
        super().__init__()
        
        self.depth = depth
        self.base_resolution = 256
        
        # Encoder
        self.encoder = UNetEncoder(
            depth, ch, norm_layer, batchnorm_from, max_channels,
            backbone_from=-1, backbone_channels=None, backbone_mode=''
        )
        
        # Decoder with image fusion
        self.decoder = UNetDecoder(
            depth, self.encoder.block_channels[:],
            norm_layer,
            attention_layer=partial(SpatialSeparatedAttention, mid_k=attention_mid_k),
            attend_from=attend_from,
            image_fusion=image_fusion  # This is the key!
        )
    
    def forward(self, image, mask, backbone_features=None):
        """
        Forward pass for iSSAM with high-res image fusion
        """
        # Get original size
        target_h, target_w = image.shape[2:]
        
        # Normalize input
        normed_image = self.normalize(image)
        x = torch.cat((normed_image, mask), dim=1)
        
        # Downsample to base resolution
        basic_input = F.interpolate(
            x, 
            size=(self.base_resolution, self.base_resolution), 
            mode='bilinear'
        ).detach()
        
        # Encoder
        intermediates = self.encoder(basic_input, backbone_features)
        
        # Decoder (get harmonized output and feature map)
        output, output_map = self.decoder(
            intermediates, 
            basic_input[:, :3, :, :],
            basic_input[:, 3:, :, :]
        )
        
        # Denormalize the low-res output
        output_lowres = self.denormalize(output)
        
        # Upsample to full resolution
        harmonized_upsampled = F.interpolate(
            output_lowres,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        )
        
        # Do image fusion at FULL resolution for sharper results
        # Compute attention map at low-res
        if hasattr(self.decoder, 'conv_attention'):
            # Decoder has attention - use it
            attention_lowres = torch.sigmoid(3.0 * self.decoder.conv_attention(output_map))
            attention_map = F.interpolate(
                attention_lowres,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            )
            # Blend at full resolution
            output_final = attention_map * image + (1.0 - attention_map) * harmonized_upsampled
        else:
            # No attention, just use upsampled result
            output_final = harmonized_upsampled
        
        return {
            'images': output_final,
            'base_images': output_final,
            'lut_images': output_final
        }


# ============================================================================
# Factory Function
# ============================================================================

def create_issam(checkpoint_path=None):
    """
    Create iSSAM model
    
    Args:
        checkpoint_path: Path to issam256.pth
    
    Returns:
        iSSAM model
    """
    model = iSSAM(
        depth=4, 
        ch=32, 
        image_fusion=True,  # Uses decoder's image fusion
        attention_mid_k=0.5,
        attend_from=2, 
        batchnorm_from=2
    )
    
    if checkpoint_path:
        print(f"Loading iSSAM: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        if len(missing) == 0 and len(unexpected) == 0:
            print("✓ iSSAM loaded perfectly!")
        else:
            print(f"⚠ Loaded with: Missing={len(missing)}, Unexpected={len(unexpected)}")
    
    return model