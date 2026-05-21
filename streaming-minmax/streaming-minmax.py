import numpy as np

def streaming_minmax_init(D):
    """
    Initialize state dict with min, max arrays of shape (D,).
    """
    # Write code here
    # pass
    res = {"min": np.full(D, np.inf),
           "max": np.full(D, -np.inf)
          }
    return res

def streaming_minmax_update(state, X_batch, eps=1e-8):
    """
    Update state's min/max with X_batch, return normalized batch.
    """
    # Write code here
    state["min"] = np.minimum(state["min"], np.min(X_batch, axis = 0, keepdims = True))
    state["max"] = np.maximum(state["max"], np.max(X_batch, axis = 0, keepdims = True))
  
    m = state["min"]
    M = state["max"]

    X_batch = (X_batch - m) / (M - m + eps)
    return X_batch
    

    