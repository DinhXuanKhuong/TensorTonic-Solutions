import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    x = np.array(x)
    sigm = 1.0 / (1 + np.exp(-1 * x)) 
    return sigm