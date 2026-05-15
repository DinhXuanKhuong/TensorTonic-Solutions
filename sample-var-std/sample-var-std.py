import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    # Write code here
    # pass
    x = np.array(x)
    n = len(x)
    mu = np.mean(x)
    var = np.sum((x - mu)**2) / (n - 1)
    std = var**0.5

    return var, std