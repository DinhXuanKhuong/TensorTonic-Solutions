import numpy as np 

def rating_normalization(matrix):
    """
    Mean-center each user's ratings in the user-item matrix.
    """
    # Write code here
    matrix = np.asarray(matrix, dtype = float)
    mask = (matrix == 0)
    
    matrix = np.where(mask, np.nan, matrix)
    
    mean_col = np.nanmean(matrix, axis = 1)

    # print(mean_col)
    res = matrix - mean_col[:, None]
    res[mask] = 0
    
    return res.tolist()