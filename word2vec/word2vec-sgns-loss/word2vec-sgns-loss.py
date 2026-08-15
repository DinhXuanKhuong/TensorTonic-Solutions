import torch
import torch.nn.functional as F

def sgns_loss(center_vec: torch.Tensor, pos_vec: torch.Tensor, neg_vecs: torch.Tensor) -> torch.Tensor:
    """
    Returns a scalar torch.Tensor: the SGNS loss.
    """
    # YOUR CODE HERE
    a = F.softplus(-center_vec @ pos_vec)
    b =  torch.sum(F.softplus( neg_vecs @(center_vec) ))
    print(a)
    print(b)
    L = a + b
    return L
