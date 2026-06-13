import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    # Write code here
    X_test = np.asarray(X_test)
    X_train = np.asarray(X_train)
    n = X_train.shape[0]
    if X_test.ndim == 2:
        d = X_test[:, None, :] -  X_train[None, :, :]
        diff = np.linalg.norm(d, axis = -1)
    elif X_test.ndim == 1:
        d = X_test[:, None] -  X_train[None, :]
        diff = d**2
    # print("hello")
    # print(d)
    # print("hello2")
    # print(diff)
    res = np.argsort(diff, axis = -1)
    # print("hello3")
    # print(res)
    
    if n < k:
        # print("pad")
        res = np.pad(res, ((0, 0), (0, k - n)), constant_values=-1)
    # print(res)
    return res[:, np.arange(k)]
    