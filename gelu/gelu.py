import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    # Write code here
    # pass
    x = np.array(x)
    # if x.ndim == 0:
    #     x = 0.5 * x * (1 + math.erf(x/np.sqrt(2)))
    # else:
    vec_erf = np.vectorize(math.erf)
    x = 0.5 * x * (1 + vec_erf(x/np.sqrt(2)))
    return x
