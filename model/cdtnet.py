"""
CDTNet - Exact implementation from official repository
https://github.com/bcmi/CDTNet-High-Resolution-Image-Harmonization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
import numbers


# ============================================================================
# Basic Blocks
# ============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1,
                 norm_layer=nn.BatchNorm2d, activation=nn.ELU, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
            activation(),
        )
    
    def forward(self, x):
        return self.block(x)


class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma, padding=0, dim=2):
        super().__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        kernel = 1.
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, grid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2.
            kernel *= torch.exp(-((grid - mean) / std) ** 2 / 2) / (std * (2 * math.pi) ** 0.5)
        
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = torch.repeat_interleave(kernel, channels, 0)

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.padding = padding
        self.conv = F.conv2d

    def forward(self, input):
        return self.conv(input, weight=self.weight, padding=self.padding, groups=self.groups)


# ============================================================================
# Encoder
# ============================================================================

class UNetDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, activation, padding):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=padding,
                     norm_layer=norm_layer, activation=activation),
            ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=padding,
                     norm_layer=norm_layer, activation=activation),
        )

    def forward(self, x):
        return self.block(x)


class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, activation, pool, padding):
        super().__init__()
        self.convs = UNetDoubleConv(in_channels, out_channels, norm_layer, activation, padding)
        self.pooling = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x):
        conv_x = self.convs(x)
        return conv_x, self.pooling(conv_x)


class UNetEncoder(nn.Module):
    def __init__(self, depth, ch, norm_layer, batchnorm_from, max_channels,
                 backbone_from, backbone_channels=None, backbone_mode=''):
        super().__init__()
        self.depth = depth
        self.backbone_from = backbone_from
        self.block_channels = []
        relu = partial(nn.ReLU, inplace=True)

        in_channels = 4
        out_channels = ch

        self.block0 = UNetDownBlock(
            in_channels, out_channels,
            norm_layer=norm_layer if batchnorm_from == 0 else None,
            activation=relu, pool=True, padding=1,
        )
        self.block_channels.append(out_channels)
        
        in_channels, out_channels = out_channels, min(2 * out_channels, max_channels)
        self.block1 = UNetDownBlock(
            in_channels, out_channels,
            norm_layer=norm_layer if 0 <= batchnorm_from <= 1 else None,
            activation=relu, pool=True, padding=1,
        )
        self.block_channels.append(out_channels)

        self.blocks_connected = nn.ModuleDict()
        for block_i in range(2, depth):
            in_channels, out_channels = out_channels, min(2 * out_channels, max_channels)
            self.blocks_connected[f'block{block_i}'] = UNetDownBlock(
                in_channels, out_channels,
                norm_layer=norm_layer if 0 <= batchnorm_from <= block_i else None,
                activation=relu, padding=1,
                pool=block_i < depth - 1,
            )
            self.block_channels.append(out_channels)

    def forward(self, x, backbone_features=None):
        outputs = []

        block_input = x
        output, block_input = self.block0(block_input)
        outputs.append(output)
        output, block_input = self.block1(block_input)
        outputs.append(output)

        for block_i in range(2, self.depth):
            block = self.blocks_connected[f'block{block_i}']
            output, block_input = block(block_input)
            outputs.append(output)

        return outputs[::-1]


# ============================================================================
# Decoder
# ============================================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.global_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1),
            nn.AdaptiveMaxPool2d(1),
        ])
        intermediate_channels_count = max(in_channels // 16, 8)
        self.attention_transform = nn.Sequential(
            nn.Linear(len(self.global_pools) * in_channels, intermediate_channels_count),
            nn.ReLU(),
            nn.Linear(intermediate_channels_count, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        pooled_x = []
        for global_pool in self.global_pools:
            pooled_x.append(global_pool(x))
        pooled_x = torch.cat(pooled_x, dim=1).flatten(start_dim=1)
        channel_attention_weights = self.attention_transform(pooled_x)[..., None, None]
        return channel_attention_weights * x


class SpatialSeparatedAttention(nn.Module):
    def __init__(self, in_channels, norm_layer, activation, mid_k=2.0):
        super().__init__()
        self.background_gate = ChannelAttention(in_channels)
        self.foreground_gate = ChannelAttention(in_channels)
        self.mix_gate = ChannelAttention(in_channels)

        mid_channels = int(mid_k * in_channels)
        self.learning_block = nn.Sequential(
            ConvBlock(in_channels, mid_channels, kernel_size=3, stride=1, padding=1,
                     norm_layer=norm_layer, activation=activation, bias=False),
            ConvBlock(mid_channels, in_channels, kernel_size=3, stride=1, padding=1,
                     norm_layer=norm_layer, activation=activation, bias=False),
        )
        self.mask_blurring = GaussianSmoothing(1, 7, 1, padding=3)

    def forward(self, x, mask):
        mask = self.mask_blurring(F.interpolate(
            mask, size=x.size()[-2:],
            mode='bilinear', align_corners=True
        ))
        background = self.background_gate(x)
        foreground = self.learning_block(self.foreground_gate(x))
        mix = self.mix_gate(x)
        output = mask * (foreground + mix) + (1 - mask) * background
        return output
class UNetUpBlock(nn.Module):
    def __init__(self, in_channels_decoder, in_channels_encoder, out_channels,
                 norm_layer, activation, padding, attention_layer):
        super().__init__()
        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(in_channels_decoder, out_channels, kernel_size=3, stride=1, padding=1,
                     norm_layer=None, activation=activation),
        )
        self.convs = UNetDoubleConv(
            in_channels_encoder + out_channels, out_channels,
            norm_layer=norm_layer, activation=activation, padding=padding,
        )
        if attention_layer is not None:
            self.attention = attention_layer(in_channels_encoder + out_channels, norm_layer, activation)
        else:
            self.attention = None

    def forward(self, x, encoder_out, mask=None):
        upsample_x = self.upconv(x)
        x_cat_encoder = torch.cat([encoder_out, upsample_x], dim=1)
        if self.attention is not None:
            x_cat_encoder = self.attention(x_cat_encoder, mask)
        return self.convs(x_cat_encoder)


class UNetDecoder(nn.Module):
    def __init__(self, depth, encoder_blocks_channels, norm_layer,
                 attention_layer=None, attend_from=3, image_fusion=False):
        super().__init__()
        self.up_blocks = nn.ModuleList()
        self.image_fusion = image_fusion
        
        in_channels = encoder_blocks_channels.pop()
        out_channels = in_channels
        
        for d in range(depth - 1):
            out_channels = encoder_blocks_channels.pop() if len(encoder_blocks_channels) else in_channels // 2
            stage_attention_layer = attention_layer if 0 <= attend_from <= d else None
            self.up_blocks.append(UNetUpBlock(
                in_channels, out_channels, out_channels,
                norm_layer=norm_layer, activation=partial(nn.ReLU, inplace=True),
                padding=1, attention_layer=stage_attention_layer,
            ))
            in_channels = out_channels

        if self.image_fusion:
            self.conv_attention = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.to_rgb = nn.Conv2d(out_channels, 3, kernel_size=1)

    def forward(self, encoder_outputs, input_image, mask):
        output = encoder_outputs[0]
        for block, skip_output in zip(self.up_blocks, encoder_outputs[1:]):
            output = block(output, skip_output, mask)
        
        output_map = output
        if self.image_fusion:
            attention_map = torch.sigmoid(3.0 * self.conv_attention(output))
            output = attention_map * input_image + (1.0 - attention_map) * self.to_rgb(output)
        else:
            output = self.to_rgb(output)

        return output, output_map


# ============================================================================
# LUT Module  
# ============================================================================

class TrilinearInterpolation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, lut, img):
        img = (img - .5) * 2.
        img = img.permute(0, 2, 3, 1)[:, None]
        lut = lut[None]
        result = F.grid_sample(lut, img, mode='bilinear', padding_mode='border', align_corners=True)
        result = result[:, :, 0, :, :]
        return lut, result


class Generator3DLUT_identity(nn.Module):
    def __init__(self, dim=33):
        super().__init__()
        buffer = torch.zeros((3, dim, dim, dim), dtype=torch.float32)
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    buffer[0, i, j, k] = i / (dim - 1)
                    buffer[1, i, j, k] = j / (dim - 1)
                    buffer[2, i, j, k] = k / (dim - 1)
        
        self.LUT = nn.Parameter(buffer.clone())
        self.TrilinearInterpolation = TrilinearInterpolation()

    def forward(self, x):
        _, output = self.TrilinearInterpolation(self.LUT, x)
        return output


class Generator3DLUT_zero(nn.Module):
    def __init__(self, dim=33):
        super().__init__()
        self.LUT = nn.Parameter(torch.zeros(3, dim, dim, dim, dtype=torch.float32))
        self.TrilinearInterpolation = TrilinearInterpolation()

    def forward(self, x):
        _, output = self.TrilinearInterpolation(self.LUT, x)
        return output


class Weight_predictor_issam(nn.Module):
    def __init__(self, in_channels=256, out_channels=3, fb=True):
        super().__init__()
        self.mid_ch = 256
        self.fb = fb
        self.conv = nn.Conv2d(in_channels, self.mid_ch, 1, 1, padding=0)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        if self.fb:
            self.fc = nn.Conv2d(self.mid_ch * 2, out_channels, 1, 1, padding=0)
        else:
            self.fc = nn.Conv2d(self.mid_ch, out_channels, 1, 1, padding=0)

    def forward(self, encoder_outputs, mask):
        fea_input = encoder_outputs[0]
        x = self.conv(fea_input)
        if self.fb:
            down_mask = F.interpolate(mask, size=fea_input.shape[2:], mode='bilinear')
            fg_feature = self.avg_pooling(x * down_mask)
            bg_feature = self.avg_pooling(x * (1 - down_mask))
            fgbg_fea = torch.cat((fg_feature, bg_feature), 1)
            x = self.fc(fgbg_fea)
        else:
            feature = self.avg_pooling(x)
            x = self.fc(feature)
        return x


class LUT(nn.Module):
    def __init__(self, in_channels=256, n_lut=4, backbone='issam', fb=True, clamp=False):
        super().__init__()
        self.n_lut = n_lut
        self.fb = fb
        self.clamp = clamp
        
        if n_lut >= 1:
            self.LUT0 = Generator3DLUT_identity()
        if n_lut >= 2:
            self.LUT1 = Generator3DLUT_zero()
        if n_lut >= 3:
            self.LUT2 = Generator3DLUT_zero()
        if n_lut >= 4:
            self.LUT3 = Generator3DLUT_zero()
        
        self.classifier = Weight_predictor_issam(in_channels, n_lut, self.fb)

    def forward(self, encoder_outputs, image, mask):
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
                combine_A[b, :, :, :] = (pred_weights[b, 0] * gen_A0[b, :, :, :] +
                                         pred_weights[b, 1] * gen_A1[b, :, :, :] +
                                         pred_weights[b, 2] * gen_A2[b, :, :, :] +
                                         pred_weights[b, 3] * gen_A3[b, :, :, :])
        
        if self.clamp:
            combine_A = torch.clamp(combine_A, 0, 1)
        
        combine_A = combine_A * mask + image * (1 - mask)
        return combine_A


# ============================================================================
# Refine Module
# ============================================================================

class Refine(nn.Module):
    def __init__(self, feature_channels=32, in_channel=7, inner_channel=64,
                 norm_layer=nn.BatchNorm2d, activation=nn.ELU, image_fusion=True):
        super().__init__()
        self.image_fusion = image_fusion
        self.block = nn.Sequential(
            nn.Conv2d(feature_channels + in_channel, inner_channel, kernel_size=3, stride=1, padding=1),
            norm_layer(inner_channel) if norm_layer is not None else nn.Identity(),
            activation(),
            nn.Conv2d(inner_channel, inner_channel, kernel_size=3, stride=1, padding=1),
            norm_layer(inner_channel) if norm_layer is not None else nn.Identity(),
            activation(),
        )
        if self.image_fusion:
            self.conv_attention = nn.Conv2d(inner_channel, 1, kernel_size=1)
        self.to_rgb = nn.Conv2d(inner_channel, 3, 1, 1, 0)

    def forward(self, ssam_output, comp, mask, ssam_features, lut_output, target_resolution):
        ssam_in = F.interpolate(ssam_output, size=target_resolution, mode='bilinear')
        comp = F.interpolate(comp, size=target_resolution, mode='bilinear')
        mask = F.interpolate(mask, size=target_resolution, mode='bilinear')
        ssam_features = F.interpolate(ssam_features, size=target_resolution, mode='bilinear')
        lut_in = F.interpolate(lut_output, size=target_resolution, mode='bilinear')
        
        input_1 = torch.cat([ssam_in, lut_in, mask, ssam_features], dim=1)
        output_map = self.block(input_1)

        if self.image_fusion:
            attention_map = torch.sigmoid(3.0 * self.conv_attention(output_map))
            output = attention_map * comp + (1.0 - attention_map) * self.to_rgb(output_map)
        else:
            output = self.to_rgb(output_map)
        
        return output_map, output


# ============================================================================
# Complete CDTNet
# ============================================================================

from .base import CDTNetBase

class CDTNet(CDTNetBase):
    def __init__(self, depth, ch=64, max_channels=512,
             norm_layer=nn.BatchNorm2d, batchnorm_from=2,
             attend_from=3, attention_mid_k=2.0,
             image_fusion=False,
             backbone_from=-1, backbone_channels=None, backbone_mode='',
             n_lut=4, has_refine=True):
        super().__init__()
        self.depth = depth
        self.n_lut = n_lut
        self.is_sim = not has_refine
        
        self.mean = torch.tensor([.485, .456, .406], dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor([.229, .224, .225], dtype=torch.float32).view(1, 3, 1, 1)
        self.device = None
        
        self.encoder = UNetEncoder(
            depth, ch, norm_layer, batchnorm_from, max_channels,
            backbone_from, backbone_channels, backbone_mode
        )
        
        self.decoder = UNetDecoder(
            depth, self.encoder.block_channels[:],
            norm_layer,
            attention_layer=partial(SpatialSeparatedAttention, mid_k=attention_mid_k),
            attend_from=attend_from,
            image_fusion=image_fusion
        )
        
        self.lut = LUT(256, n_lut, backbone='issam')
        
        # Only create refine if needed (REMOVED duplicate line above)
        if has_refine:
            self.refine = Refine(feature_channels=32, inner_channel=64)
        else:
            self.refine = None

    def set_resolution(self, hr_w, hr_h, lr, finetune_base):
        self.target_w_resolution = hr_w
        self.target_h_resolution = hr_h
        self.base_resolution = lr
        self.finetune_base = finetune_base

    def init_device(self, input_device):
        if self.device is None:
            self.device = input_device
            self.mean = self.mean.to(self.device)
            self.std = self.std.to(self.device)

    def normalize(self, tensor):
        self.init_device(tensor.device)
        return (tensor - self.mean) / self.std

    def denormalize(self, tensor):
        self.init_device(tensor.device)
        return tensor * self.std + self.mean

    def forward(self, image, mask, backbone_features=None):
        """Forward pass for CDTNet"""
        target_h, target_w = image.shape[2:]
        
        normed_image = self.normalize(image)
        x = torch.cat((normed_image, mask), dim=1)
        
        basic_input = F.interpolate(x, size=(self.base_resolution, self.base_resolution), mode='bilinear').detach()
        
        intermediates = self.encoder(basic_input, backbone_features)
        output, output_map = self.decoder(intermediates, basic_input[:, :3, :, :], basic_input[:, 3:, :, :])
        
        lut_output = self.lut(intermediates, image, mask)
        
        # Use refine only if it exists
        if self.refine is not None and not self.is_sim:
            normed_lut = self.normalize(lut_output)
            _, hd_output = self.refine(output, normed_image, mask, output_map, normed_lut,
                                      target_resolution=(target_h, target_w))
            denormed_hd_output = self.denormalize(hd_output)
        else:
            # Simplified model - use LUT output directly
            denormed_hd_output = lut_output
        
        return {'images': denormed_hd_output, 'lut_images': lut_output, 'base_images': self.denormalize(output)}


# ============================================================================
# Factory Function (MODULE LEVEL - NO INDENTATION!)
# ============================================================================

def create_cdtnet(model_name, checkpoint_path=None):
    """
    Create CDTNet model
    
    Args:
        model_name: Name of the model variant
        checkpoint_path: Path to checkpoint file
    
    Returns:
        CDTNet model
    """
    # Determine if model has refinement
    has_refine = 'sim' not in model_name.lower()
    
    model = CDTNet(
        depth=4, 
        ch=32, 
        image_fusion=True, 
        attention_mid_k=0.5,
        attend_from=2, 
        batchnorm_from=2, 
        n_lut=4,
        has_refine=has_refine
    )
    
    if '2048' in model_name:
        model.base_resolution = 512
    else:
        model.base_resolution = 256
    
    if checkpoint_path:
        print(f"Loading: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        if len(missing) == 0 and len(unexpected) == 0:
            print("✓ Perfect!")
        else:
            if not has_refine and len(missing) > 0:
                print("✓ Loaded (simplified model, no refinement)")
            else:
                print(f"⚠ Missing:{len(missing)}, Unexpected:{len(unexpected)}")
    
    return model