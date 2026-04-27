import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    # pass
    d_k = K.shape[-1]

    M = torch.matmul(Q, K.transpose(1, 2))
    M /= math.sqrt(d_k)

    M = F.softmax(M, dim = 2)
    
    score = torch.matmul(M, V)

    return score