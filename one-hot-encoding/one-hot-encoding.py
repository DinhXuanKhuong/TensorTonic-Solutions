import numpy as np

def one_hot(y, num_classes=None):
    """
    Convert integer labels y ∈ {0,...,K-1} into one-hot matrix of shape (N, K).
    """
    # Write code here
    y = np.array(y)

    if num_classes is None:
        num_classes = max(y) + 1

    n = len(y)

    res = np.zeros((n, num_classes))

    res[np.arange(n), y] = 1

    return res

    