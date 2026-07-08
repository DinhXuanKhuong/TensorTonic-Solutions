import numpy as np 
def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    # Write code here
    X = np.asarray(X)
    y = np.asarray(y)
    w = np.linalg.inv(X.T @ X + lam * np.eye(len(X[0]))) @ (X.T @ y)
    return w