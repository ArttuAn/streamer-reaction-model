"""Load tensor checkpoints without allowing pickled Python code to execute."""
import torch

def load_tensor_data(path, *, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=True)
