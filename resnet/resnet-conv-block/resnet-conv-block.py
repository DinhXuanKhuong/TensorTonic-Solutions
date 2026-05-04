import numpy as np

def conv_block(x, W1, W2, Ws):
    """
    Returns: np.ndarray with sum of main path output and projected shortcut
    """
    # YOUR CODE HERE
    # pass
    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    
    short_cut = x @ Ws
    
    h = np.maximum(0, x @ W1)

    z = np.maximum(0, h @ W2 + short_cut) 

    # out = z + short_cut

    return z