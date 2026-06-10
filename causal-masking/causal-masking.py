import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: float used to mask future positions (e.g., -1e9)
    Return: masked scores (same shape, dtype=float)
    """
    # Write code here
    scores = np.asarray(scores)
    masked = scores.copy()

    T = scores.shape[-1]

    i = np.arange(T)[:, None]
    j = np.arange(T)

    mask = j > i

    masked = np.where(mask, mask_value, masked)
    return masked
    
    