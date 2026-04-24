import numpy as np
    
def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    # Write code here
    # pass
    matrix = np.array(matrix)
    if matrix.ndim != 2:
        return None
    if norm_type == 'l2':
        try:
            divided = np.linalg.norm(matrix, axis=axis, keepdims = True)
        except Exception:
            return None
    elif norm_type == "l1":
        try:
            divided = np.sum(np.abs(matrix), axis=axis, keepdims = True)
        except Exception:
            return None
        
    elif norm_type == "max":
        try:
            divided = np.max(np.abs(matrix), axis = axis, keepdims = True)
        except Exception:
            return None
    else:
        return None

    k = np.divide(matrix, divided)
    k = np.where(np.isnan(k) | np.isinf(k), 0, k)
    return k
    

    