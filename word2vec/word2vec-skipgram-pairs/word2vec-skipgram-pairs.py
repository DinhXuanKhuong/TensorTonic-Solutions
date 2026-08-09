import torch

def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
    """
    Returns int64 torch.Tensor of shape (num_pairs, 2).
    """
    # YOUR CODE HERE
    n = len(token_ids)
    res = []
    for i in range(n):
        l = max(0, i - window)
        r = min(n - 1, i + window)
        for j in range(l, r + 1):
            if j == i: 
                continue
            res.append([token_ids[i], token_ids[j]])
    if len(res) == 0:
        return torch.empty((0, 2), dtype=torch.int64)
    return torch.tensor(res, dtype = torch.int64)