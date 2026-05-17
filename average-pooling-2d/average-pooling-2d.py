import numpy as np 


def average_pooling_2d(X, pool_size):
    """
    Apply 2D average pooling with non-overlapping windows.
    """
    # Write code here

    X = np.array(X)

    h, w = X.shape

    h_out = h // pool_size
    w_out = w // pool_size

    res = np.zeros((h_out, w_out))

    for i in range(h_out):
        for j in range(w_out):
            arr = X[i * pool_size: (i + 1) * pool_size, j * pool_size: (j + 1) * pool_size]
            res[i, j] = np.mean(arr)
    return res.tolist()
    
    