import numpy as np

def get_alpha_bar(betas):
    """
    Compute cumulative product of (1 - beta).
    Returns list of floats rounded to 6 decimals.
    """
    # YOUR CODE HERE
    betas = np.asarray(betas.copy(), dtype = float)
    betas[0] = 1 - betas[0]
    for i in range(1, len(betas)):
        betas[i] = (1 - betas[i]) *  betas[i - 1]
        
    return np.round(betas, 6).tolist()
    

def forward_diffusion(x_0, t, betas, epsilon):
    """
    Returns: tuple of (np.ndarray x_t, np.ndarray epsilon) with same shape as x_0
    """
    # YOUR CODE HERE
    x_0 = np.asarray(x_0, dtype = np.float64)
    epsilon = np.asarray(epsilon, dtype = np.float64)
    
    alpha_bar = get_alpha_bar(betas)

    x_t = np.sqrt(alpha_bar[t-1]) * x_0 + np.sqrt(1 - alpha_bar[t-1]) * epsilon
    return np.round(x_t, 4).tolist()