import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
    # Write code here
    X = np.asarray(X, dtype=float)

    # 1. Center each feature
    X_centered = X - np.mean(X, axis=0)

    # 2. Covariance matrix
    n = X.shape[0]
    cov = (X_centered.T @ X_centered) / (n - 1)

    # 3. Standard deviation of each feature
    std = np.sqrt(np.diag(cov))

    # 4. sigma sigma^T
    denominator = np.outer(std, std)

    # 5. Pearson correlation matrix
    R = cov / denominator

    return R