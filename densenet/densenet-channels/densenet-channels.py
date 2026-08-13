import math
import torch

def densenet_channel_counts(stem_channels: int, growth_rate: int, block_layers, compression: float) -> torch.Tensor:
    """
    Returns a 1D int64 torch.Tensor of channel counts at each stage.
    """
    # YOUR CODE HERE
    res = []
    res.append(stem_channels)
    for i, val in enumerate(block_layers):
    
        res.append(val * growth_rate + res[-1])
        res.append(int(res[-1] * compression))
    
    return torch.tensor(res[:(len(res)  - 1)])
        