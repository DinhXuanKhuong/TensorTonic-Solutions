import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    # Write code here
    x = np.array(x)
    res = np.where(x > 0, x, 0)
    return res