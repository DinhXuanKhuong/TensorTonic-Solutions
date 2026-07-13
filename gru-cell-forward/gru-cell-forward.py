import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    # Write code here
    x_new, _ = _as2d(x, -1)
    h_prev_new, _ = _as2d(h_prev, -1)
    
    Uh = params.get("Uh", None)
    Ur = params.get("Ur", None)
    Uz = params.get("Uz", None)
    Wh = params.get("Wh", None)
    Wr = params.get("Wr", None)
    
    Wz = params.get("Wz", None)
    bh = params.get("bh", None)
    br = params.get("br", None)
    bz = params.get("bz", None)

    
    zt = _sigmoid(x_new @ Wz + h_prev_new @ Uz + bz)
    rt = _sigmoid(x_new @ Wr + h_prev_new @ Ur + br)

    ht_nga = np.tanh(x_new @ Wh + (rt * h_prev_new) @ Uh + bh)

    ht = (1 - zt) * h_prev_new + zt * ht_nga

    return np.squeeze(ht)
    
    
    