import numpy as np

def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns: np.ndarray of same shape as input with batch-normalized and skip-connected output
    """
    # YOUR CODE HERE

    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)

    gamma1 = np.array(gamma1)
    gamma2 = np.array(gamma2)

    beta1 = np.array(beta1)
    beta2 = np.array(beta2)

    
    eps = 1e-5
    if mode == "post":
        skip_con = x.copy()
        
        x = x @ W1
        
        mu1 = np.mean(x, axis = 0, keepdims = True)
        std1 = np.var(x, axis = 0, keepdims = True)
        
        x_normed = (x - mu1) / np.sqrt(std1 + eps)
        
        x_normed = gamma1 * x_normed + beta1
        x_normed = np.maximum(0, x_normed) 
        
        x_normed = x_normed @ W2
        

        mu2 = np.mean(x_normed, axis = 0, keepdims = True)
        std2 = np.var(x_normed, axis = 0, keepdims = True)
        
        x_normed2 = (x_normed - mu2) / np.sqrt(std2 + eps)

        x_normed2 = gamma2 * x_normed2 + beta2
        
        x_normed2 += skip_con
        
        x_normed2 = np.maximum(0, x_normed2)
        
        # return x_normed2 
        return {"output" : np.round(x_normed2, 4), "mode" : mode} 
        # return np.round(x_normed2, 4) 
        
        
    elif mode == "pre":
        skip_con = x.copy()
        
        mu1 = np.mean(x, axis = 0, keepdims = True)
        std1 = np.var(x, axis = 0, keepdims = True)
        x_normed = (x - mu1) / np.sqrt(std1 + eps)
        
        x_normed = gamma1 * x_normed + beta1
        x_normed = np.maximum(0, x_normed) 
        x_normed = x_normed @ W1

        mu2 = np.mean(x_normed, axis = 0, keepdims = True)
        std2 = np.var(x_normed, axis = 0, keepdims = True)
        x_normed2 = (x_normed - mu2) / np.sqrt(std2 + eps)

        x_normed2 = gamma2 * x_normed2 + beta2
        
        x_normed2 = np.maximum(0, x_normed2)

        x_normed2 = x_normed2 @ W2

        x_normed2 += skip_con
        
        return {"output" : np.round(x_normed2, 4), "mode" : mode} 
    else:
        return None