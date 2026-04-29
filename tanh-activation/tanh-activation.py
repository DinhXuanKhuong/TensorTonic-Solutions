import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # Write code here
    # pass
    x = np.array(x)
    e_x = np.exp(x)
    e_minus_x = np.exp(-x)

    res = (e_x - e_minus_x) / (e_x + e_minus_x)
    return res