import numpy as np 
def min_max_scaling(data):
    """
    Scale each column of the data matrix to the [0, 1] range.
    """
    # Write code here
    data = np.array(data)
    min_col = np.min(data, axis = 0, keepdims = True)
    max_col = np.max(data, axis = 0, keepdims = True)
    deno = max_col - min_col
    deno = np.where(deno == 0, 1, deno)
    return ((data - min_col) / (deno)).tolist()
    