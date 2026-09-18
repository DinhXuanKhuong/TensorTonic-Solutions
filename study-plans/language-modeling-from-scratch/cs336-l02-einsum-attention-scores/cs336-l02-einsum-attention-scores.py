import torch

def attention_scores(q, k, num_heads):
    """
    Returns: tensor of shape (batch, heads, query_length, key_length)
    """
    B, S_q, D = q.shape
    _, S_k, _ = k.shape
    
    d_h = D // num_heads
    d_h = torch.tensor(d_h)
    
    q = q.reshape(B, S_q, num_heads, -1)
    k = k.reshape(B, S_k, num_heads, -1)

    q = torch.moveaxis(q, 2, 1)
    k = torch.moveaxis(k, 2, 1)
    
    A = (q @ k.transpose(2,3)) / (torch.sqrt(d_h))
    return A
