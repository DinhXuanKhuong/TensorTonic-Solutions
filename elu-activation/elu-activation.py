import numpy as np
def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    # Write code here
    x = np.array(x)
    res = np.where(x > 0, x, alpha * (np.exp(x) - 1))
    return list(res)