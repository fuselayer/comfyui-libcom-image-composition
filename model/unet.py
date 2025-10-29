
import torch
from torch import nn as nn
import torch.nn.functional as F
from functools import partial

from .basic_blocks import ConvBlock
from .ops import FeaturesConnector

class UNetEncoder(nn.Module):
    def __init__(self, depth, ch, norm_layer, batchnorm_from, max_channels):
        super(UNetEncoder, self).__init__()
        self.depth = depth
        self.block_channels = []

        relu = partial(nn.ReLU, inplace=True)

        in_channels = 4  # composite_image (3) + mask (1)
        out_channels = ch
        self.block0 = UNetDownBlock(
            in_channels, out_channels,
            norm_layer=norm_layer if batchnorm_from == 0 else None,
            activation=relu,
            pool=True,
            padding=1,
        )
        self.block_channels.append(out_channels)

        in_channels, out_channels = out_channels, min(2 * out_channels, max_channels)
        self.block1 = UNetDownBlock(
            in_channels, out_channels,
            norm_layer=norm_layer if 0 <= batchnorm_from <= 1 else None,
            activation=relu,
            pool=True,
            padding=1,
        )
        self.block_channels.append(out_channels)

        self.blocks_connected = nn.ModuleDict()
        for block_i in range(2, depth):
            in_channels, out_channels = out_channels, min(2 * out_channels, max_channels)
            self.blocks_connected[f'block{block_i}'] = UNetDownBlock(
                in_channels, out_channels,
                norm_layer=norm_layer if 0 <= batchnorm_from <= block_i else None,
                activation=relu,
                padding=1,
                pool=block_i < depth - 1,
            )
            self.block_channels.append(out_channels)

    def forward(self, x):
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

class UNetDecoder(nn.Module):
    def __init__(self, depth, encoder_blocks_channels, norm_layer, attention_layer=None, attend_from=3, image_fusion=True):
        super(UNetDecoder, self).__init__()
        self.up_blocks = nn.ModuleList()
        self.image_fusion = image_fusion

        in_channels = encoder_blocks_channels.pop()
        out_channels = in_channels
        for d in range(depth - 1):
            out_channels = encoder_blocks_channels.pop() if len(encoder_blocks_channels) else in_channels // 2
            stage_attention_layer = attention_layer if 0 <= attend_from <= d else None
            self.up_blocks.append(UNetUpBlock(
                in_channels, out_channels, out_channels,
                norm_layer=norm_layer,
                activation=partial(nn.ReLU, inplace=True),
                padding=1,
                attention_layer=stage_attention_layer,
            ))
            in_channels = out_channels

        # Refinement layers with corrected channel count
        self.block = nn.Sequential(
            nn.Conv2d(out_channels + 7, 32, kernel_size=3, stride=1, padding=1),
            norm_layer(32) if norm_layer is not None else nn.Identity(),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            norm_layer(32) if norm_layer is not None else nn.Identity(),
            nn.ELU(),
        )
        self.conv_attention = nn.Conv2d(32, 1, kernel_size=1)
        self.to_rgb = nn.Conv2d(32, 3, 1, 1, 0)

    def forward(self, encoder_outputs, input_image, mask, lut_output, target_resolution):
        output = encoder_outputs[0]
        for block, skip_output in zip(self.up_blocks, encoder_outputs[1:]):
            output = block(output, skip_output, mask)

        # Refinement logic is now part of the decoder's forward pass
        ssam_in = F.interpolate(output, size=target_resolution, mode='bilinear', align_corners=False)
        comp = F.interpolate(input_image, size=target_resolution, mode='bilinear', align_corners=False)
        mask_interp = F.interpolate(mask, size=target_resolution, mode='bilinear', align_corners=False)
        ssam_features = F.interpolate(output, size=target_resolution, mode='bilinear', align_corners=False)
        lut_in = F.interpolate(lut_output, size=target_resolution, mode='bilinear', align_corners=False)

        input_1 = torch.cat([ssam_in, lut_in, mask_interp, comp], dim=1)
        output_map = self.block(input_1)

        attention_map = torch.sigmoid(3.0 * self.conv_attention(output_map))
        final_output = attention_map * comp + (1.0 - attention_map) * self.to_rgb(output_map)

        return final_output

class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, activation, pool, padding):
        super(UNetDownBlock, self).__init__()
        self.convs = UNetDoubleConv(
            in_channels, out_channels,
            norm_layer=norm_layer,
            activation=activation,
            padding=padding,
        )
        self.pooling = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x):
        conv_x = self.convs(x)
        return conv_x, self.pooling(conv_x)

class UNetUpBlock(nn.Module):
    def __init__(self, in_channels_decoder, in_channels_encoder, out_channels, norm_layer, activation, padding, attention_layer):
        super(UNetUpBlock, self).__init__()
        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(
                in_channels_decoder, out_channels,
                kernel_size=3, stride=1, padding=1,
                norm_layer=None, activation=activation,
            )
        )

        self.convs = UNetDoubleConv(
            in_channels_encoder + out_channels, out_channels,
            norm_layer=norm_layer,
            activation=activation,
            padding=padding,
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

class UNetDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, activation, padding):
        super(UNetDoubleConv, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(
                in_channels, out_channels,
                kernel_size=3, stride=1, padding=padding,
                norm_layer=norm_layer, activation=activation,
            ),
            ConvBlock(
                out_channels, out_channels,
                kernel_size=3, stride=1, padding=padding,
                norm_layer=norm_layer, activation=activation,
            ),
        )

    def forward(self, x):
        return self.block(x)
