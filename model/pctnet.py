"""
PCTNet - Wrapper for the libcom implementation
https://github.com/bcmi/libcom
"""

import torch
from libcom.image_harmonization import ImageHarmonizationModel

def create_pctnet(checkpoint_path=None):
    """
    Create PCTNet model using the libcom library.

    Args:
        checkpoint_path: Path to the checkpoint file.
                         (Note: libcom handles its own model downloading,
                          but we can pass the path for consistency)

    Returns:
        PCTNet model
    """
    # libcom's ImageHarmonizationModel handles model downloading and loading internally.
    # We instantiate it with model_type='PCTNet' to get the correct model.
    # The device is set to 'cpu' for now, it will be moved to the correct device later.
    harmonization_model = ImageHarmonizationModel(device='cpu', model_type='PCTNet')

    # The actual nn.Module is stored in the 'model' attribute of the ImageHarmonizationModel
    model = harmonization_model.model

    # The libcom model expects the checkpoint to be in a specific location.
    # We will load the state dict here to ensure it's using the one from our checkpoints folder.
    if checkpoint_path:
        print(f"Loading: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        if len(missing) == 0 and len(unexpected) == 0:
            print("✓ Perfect!")
        else:
            print(f"⚠ Missing:{len(missing)}, Unexpected:{len(unexpected)}")

    return model
