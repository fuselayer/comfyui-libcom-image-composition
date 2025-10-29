"""
Detailed checkpoint inspection to understand exact architecture
"""

import torch

checkpoint_path = 'CDTNet_iHarmony4_256.pth'

print("="*80)
print(f"Analyzing: {checkpoint_path}")
print("="*80)

state_dict = torch.load(checkpoint_path, map_location='cpu')

# Analyze encoder structure
print("\n" + "="*80)
print("ENCODER STRUCTURE")
print("="*80)
encoder_keys = [k for k in state_dict.keys() if k.startswith('encoder.')]
print(f"Total encoder keys: {len(encoder_keys)}\n")

# Group by block
for block_name in ['block0', 'block1', 'blocks_connected.block0', 'blocks_connected.block1', 
                    'blocks_connected.block2', 'blocks_connected.block3']:
    block_keys = [k for k in encoder_keys if block_name in k]
    if block_keys:
        print(f"\n{block_name}:")
        for k in block_keys[:5]:  # First 5 keys
            print(f"  {k}: {state_dict[k].shape}")

# Analyze decoder structure
print("\n" + "="*80)
print("DECODER STRUCTURE")
print("="*80)
decoder_keys = [k for k in state_dict.keys() if k.startswith('decoder.')]
print(f"Total decoder keys: {len(decoder_keys)}\n")

# Analyze each up_block
for i in range(4):
    print(f"\nup_blocks.{i}:")
    block_keys = [k for k in decoder_keys if f'up_blocks.{i}.' in k]
    for k in sorted(block_keys)[:8]:  # First 8 keys
        print(f"  {k}: {state_dict[k].shape}")

# Final decoder layers
print("\nFinal decoder layers:")
for k in decoder_keys:
    if 'conv_attention' in k or 'to_rgb' in k:
        print(f"  {k}: {state_dict[k].shape}")

# Analyze LUT structure
print("\n" + "="*80)
print("LUT STRUCTURE")
print("="*80)
lut_keys = [k for k in state_dict.keys() if k.startswith('lut.')]
for k in sorted(lut_keys):
    print(f"  {k}: {state_dict[k].shape}")

# Analyze refine structure
print("\n" + "="*80)
print("REFINE STRUCTURE")
print("="*80)
refine_keys = [k for k in state_dict.keys() if k.startswith('refine.')]
print(f"Total refine keys: {len(refine_keys)}\n")
for k in sorted(refine_keys):
    print(f"  {k}: {state_dict[k].shape}")

# Determine base channels
print("\n" + "="*80)
print("ARCHITECTURE ANALYSIS")
print("="*80)

# Look at first encoder conv to determine base channels
first_conv_key = 'encoder.block0.convs.block.0.block.0.weight'
if first_conv_key in state_dict:
    shape = state_dict[first_conv_key].shape
    base_ch = shape[0]  # Output channels
    print(f"First encoder conv: {shape}")
    print(f"Inferred base_channels: {base_ch}")

# Look at decoder final layer
final_key = 'decoder.to_rgb.weight'
if final_key in state_dict:
    shape = state_dict[final_key].shape
    final_ch = shape[1]  # Input channels
    print(f"Decoder final conv input: {final_ch} channels")
    print(f"This confirms base_channels: {final_ch}")

# Check if there's an attention module in up_blocks.2
attention_keys = [k for k in decoder_keys if 'attention' in k and 'up_blocks.2' in k]
if attention_keys:
    print(f"\nup_blocks.2 has ATTENTION module with {len(attention_keys)} parameters")
    print("Sample attention keys:")
    for k in attention_keys[:5]:
        print(f"  {k}")

print("\n" + "="*80)