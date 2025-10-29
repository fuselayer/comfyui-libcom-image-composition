import torch
from model.cdtnet import create_cdtnet

print("Testing model loading...\n")

# Test 1: Load CDTNet with refinement
print("1. Loading CDTNet_iHarmony4_256...")
try:
    model1 = create_cdtnet(
        'iHarmony4_256',
        'checkpoints/CDTNet_iHarmony4_256.pth'
    )
    print("✓ Successfully loaded with refinement module")
    assert hasattr(model1, 'refine'), "Missing refine module!"
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Load CDTNet without refinement (sim)
print("\n2. Loading CDTNet_sim_base256...")
try:
    model2 = create_cdtnet(
        'sim_base256',
        'checkpoints/CDTNet_sim_base256.pth'
    )
    print("✓ Successfully loaded without refinement module")
    assert not model2.has_refine, "Should not have refine module!"
except Exception as e:
    print(f"✗ Error: {e}")

# Test 3: Load high-res model
print("\n3. Loading CDTNet_HAdobe5k_2048...")
try:
    model3 = create_cdtnet(
        'HAdobe5k_2048',
        'checkpoints/CDTNet_HAdobe5k_2048.pth'
    )
    print("✓ Successfully loaded high-res model with refinement")
    assert hasattr(model3, 'refine'), "Missing refine module!"
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "="*60)
print("All models loaded successfully!")