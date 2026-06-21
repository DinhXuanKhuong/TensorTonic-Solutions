import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    """
    Perform one Nesterov Momentum update step.
    """
    # Write code here
    # pass
    w = np.asarray(w, dtype = np.float64)
    v = np.asarray(v, dtype = np.float64)
    grad = np.asarray(grad, dtype = np.float64)
    
    # w_look = w - momentum * v 
    v_new =  lr * grad  + momentum * v 
    
    w_new = w - v_new
    return w_new, v_new