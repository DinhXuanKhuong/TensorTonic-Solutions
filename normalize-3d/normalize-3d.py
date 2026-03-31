import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    # Your code here
    v = np.array(v)

    if v.ndim == 1:
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        v_norm = v / norm 
    else:
        norm = np.linalg.norm(v, axis = 1, keepdims = True)
        norm[norm == 0] = 1
        v_norm = v / norm
    return v_norm
        