import torch

def rmsnorm(x, g, epsilon):
    """
    Returns: RMS-normalized tensor
    """
    res = x / torch.sqrt((torch.mean(x**2, dim = -1, keepdims = True) + epsilon)) * g 
    return res
