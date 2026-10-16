"""Load tensor checkpoints without allowing pickled Python code to execute."""
import torch
from packaging.version import Version

def load_tensor_data(path, *, map_location="cpu"):
    # Older weights-only unpicklers have known arbitrary-code-execution bypasses.
    if Version(torch.__version__.split("+")[0]) < Version("2.14.1"):
        raise RuntimeError("Upgrade PyTorch to 2.14.1 or newer before loading checkpoints.")
    return torch.load(path, map_location=map_location, weights_only=True)
