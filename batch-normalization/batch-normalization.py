import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    # Write code here
    # pass
    x = np.array(x)
    gamma = np.array(gamma)
    beta = np.array(beta)
    
    dim = x.ndim

    if dim == 2:
        mu = np.mean(x, axis = 0, keepdims = True)
        var = np.var(x, axis = 0, keepdims = True)        
    elif dim == 4:
        mu = np.mean(x, axis = (0, 2, 3), keepdims = True)
        var = np.var(x, axis = (0, 2, 3), keepdims = True)
        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)
    else:
        return None
    
    x_normed = (x - mu) / np.sqrt(var + eps)
    res = gamma * x_normed + beta
    
    return res