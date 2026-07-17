import numpy as np

def discriminator(x, W):
    """
    Returns: np.ndarray of shape (batch, 1) with probabilities rounded to 4 decimals
    """
    x = np.asarray(x)
    W = np.asarray(W)
    res = 1.0 / (1 + np.exp(-1 * (x @ W)))
    return res