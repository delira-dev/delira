import os
if "torch" in os.environ["DELIRA_BACKEND"]:
    from .torch import save_checkpoint as torch_save_checkpoint
    from .torch import load_checkpoint as torch_load_checkpoint

if "tf" in os.environ["DELIRA_BACKEND"]:
    from .tf import save_checkpoint, load_checkpoint
