import numpy as np

def make_diagonal(v):
    """
    Returns: (n, n) NumPy array with v on the main diagonal
    """
    # Write code here
    # pass
    v = np.array(v)
    n = len(v)
    res = np.zeros((n,n))

    for i in range(n):
        res[i, i] = v[i]
    return res
