import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    """
    Scale X to [0,1]. If 2D and axis=0 (default), scale per column.
    Return np.ndarray (float).
    """
    # Write code here
    X = np.array(X)

    if X.ndim == 1:
        X = (X - np.min(X)) / (np.max(X) - np.min(X) + eps)
    else:
        min_col = np.min(X, axis = axis, keepdims = True)
        max_col = np.max(X, axis = axis, keepdims = True)
        X = (X - min_col) / (max_col  - min_col + eps)
    return X
        
    
    