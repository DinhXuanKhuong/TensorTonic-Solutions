import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    # Your code here
    v = np.array(v)
    w = np.array(w)

    norm_v = np.linalg.norm(v)
    norm_w = np.linalg.norm(w)
    if norm_v == 0 or norm_w == 0:
        return np.nan
    
    cosin = (v @ w) / ( norm_v * norm_w)

    return np.arccos(cosin)