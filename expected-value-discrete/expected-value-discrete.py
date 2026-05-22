import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x = np.asarray(x)
    p = np.asarray(p)

    if (not np.isclose(sum(p), 1, 1e-6)) or len(x) != len(p):
        raise ValueError
    
    return sum (x * p)