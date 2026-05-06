import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns: np.ndarray with bottleneck residual block output (compress, process, expand + skip)
    """
    # YOUR CODE HERE
    # pass
    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    W3 = np.array(W3)
    Ws = np.array(Ws)

    # print(W1.shape, W2.shape, W3.shape, Ws.shape)

    short_cut = x @ Ws

    x1 = np.maximum(0, x @ W1)
    x2 = np.maximum(0, x1 @ W2)
    x3 =  x2 @ W3

    out = np.maximum(0, x3 + short_cut)
    return out
