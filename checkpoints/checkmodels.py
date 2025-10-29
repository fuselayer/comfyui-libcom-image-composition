import torch

# Load and inspect each checkpoint
checkpoints = [
    'CDTNet_iHarmony4_256.pth',
    'CDTNet_HAdobe5k_2048.pth', 
    'CDTNet_sim_base256.pth'
]

for ckpt_path in checkpoints:
    print(f"\n{'='*60}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*60}")
    
    state_dict = torch.load(ckpt_path, map_location='cpu')
    
    # Group keys by top-level module
    modules = {}
    for key in state_dict.keys():
        top_level = key.split('.')[0]
        if top_level not in modules:
            modules[top_level] = []
        modules[top_level].append(key)
    
    # Print structure
    for module, keys in sorted(modules.items()):
        print(f"\n{module}: {len(keys)} parameters")
        print(f"  First few keys: {keys[:3]}")
        print(f"  Last few keys: {keys[-3:]}")