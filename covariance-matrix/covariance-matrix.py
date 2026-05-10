import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    # pass
    X = np.array(X)
    N = len(X)
    if X.ndim != 2 or N < 2:
        return None
    mu = np.mean(X, axis = 0, keepdims = True)
    X_centered = X - mu


    res = (1 / (N - 1)) * X_centered.T @ X_centered

    return res