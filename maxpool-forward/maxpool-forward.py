import numpy as np 



def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    # Write code here

    X = np.array(X)
    h, w = X.shape

    h_out = (h - pool_size) // stride + 1
    w_out = (w - pool_size) // stride + 1

    res = np.zeros((h_out, w_out))

    for i in range(h_out):
        for j in range(w_out):
            arr = X[i * stride: i * stride + pool_size, j * stride: j * stride + pool_size]
            res[i, j] = np.max(arr)
    return res.tolist()