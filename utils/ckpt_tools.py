import torch

def inspect_state_dict(ckpt_path: str, max_keys: int = 20):
    """
    Print top-level structure and a sample of keys in a checkpoint.
    Helps determine if it's wrapped (e.g., 'state_dict') or has 'module.' prefixes.
    """
    state = torch.load(ckpt_path, map_location='cpu')
    print("Type:", type(state))
    if isinstance(state, dict):
        print("Top-level keys:", list(state.keys())[:10])
        if 'state_dict' in state and isinstance(state['state_dict'], dict):
            keys = list(state['state_dict'].keys())
            print("state_dict keys sample ({0} total):".format(len(keys)), keys[:max_keys])
        else:
            keys = list(state.keys())
            print("flat dict keys sample ({0} total):".format(len(keys)), keys[:max_keys])
            mod_pref = any(k.startswith('module.') for k in keys)
            print("Has 'module.' prefix:", mod_pref)
    else:
        try:
            keys = list(state.keys())
            print("Unknown mapping keys sample ({0} total):".format(len(keys)), keys[:max_keys])
        except Exception:
            print("Checkpoint is not a mapping (unexpected format).")