import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    # Write code here
    X = np.asarray(X, dtype= float)
    res = X.copy()
    if strategy == "mean":
        col = np.nanmean(X, axis = 0)
    else:
        col = np.nanmedian(X, axis = 0)
    
    col = np.where(np.isnan(col), 0, col)
    
    res_filled = np.where(np.isnan(res), col, res)

    
    return res_filled