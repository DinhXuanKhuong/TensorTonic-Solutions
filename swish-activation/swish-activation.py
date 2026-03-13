import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))

def swish(x):
    """
    Implement Swish activation function.
    """
    # Write code here
    x = np.array(x)
    return x * sigmoid(x)

    