import torch

def subsample_keep_probs(counts: torch.Tensor, t: float = 1e-5) -> torch.Tensor:
    """
    Returns torch.Tensor of shape (vocab_size,) with the keep-probability for each word.
    """
    # YOUR CODE HERE
    N = torch.sum(counts)
    P = torch.minimum(torch.tensor(1), torch.sqrt(t / (counts / N)))
    return P
