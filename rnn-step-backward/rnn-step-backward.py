import numpy as np

def rnn_step_backward(dh, cache):
    """
    Returns:
        dx_t: gradient wrt input x_t      (shape: D,)
        dh_prev: gradient wrt previous h (shape: H,)
        dW: gradient wrt W               (shape: H x D)
        dU: gradient wrt U               (shape: H x H)
        db: gradient wrt bias            (shape: H,)
    """
    # Write code here
    x_t = np.asarray(cache[0])
    h_prev = np.asarray(cache[1])
    h_t = np.asarray(cache[2])
    W = np.asarray(cache[3])
    U = np.asarray(cache[4])
    b = np.asarray(cache[5])
    
    d_tanh = dh * (1 - h_t**2)
    
    dx_t = W.T @ d_tanh
    dh_prev = U.T @ d_tanh
    
    dW =  np.outer(d_tanh, x_t)
    dU =  np.outer(d_tanh, h_prev)
    
    db = d_tanh

    return dx_t, dh_prev, dW, dU, db
