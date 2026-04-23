import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # Write code here
    # pass
    x = np.array(x)

    dim = x.ndim
    if (dim == 3):
        # c, h, w = x.shape
        res = np.mean(x, axis = (1, 2))
    else:
        # n, c, h, w = x.shape
        res = np.mean(x, axis = (2, 3))
    return res
        