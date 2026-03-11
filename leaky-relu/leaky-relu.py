import numpy as np


def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    # Write code here
    # pass
    x = np.array(x)
    x = np.where(x < 0, alpha * x, x)
    return x