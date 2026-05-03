import numpy as np

def vgg_maxpool(x: np.ndarray) -> np.ndarray:
    """
    Implement VGG-style max pooling (2x2, stride 2).
    """
    # Your implementation here
    N, H, W, C = x.shape

    H_new, W_new = H // 2, W // 2

    res = np.zeros((N, H_new, W_new, C))

    for n in range(N):
        for i in range(H_new):
            for j in range(W_new):
                for c in range(C):
                    res[n, i, j, c] = max(x[n, 2 * i, 2 * j, c], x[n, 2*i, 2*j + 1, c], x[n, 2 * i + 1, 2*j, c], x[n, 2*i + 1, 2 * j + 1, c])

    return res

    