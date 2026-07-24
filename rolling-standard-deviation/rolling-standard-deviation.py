import numpy as np
def rolling_std(values, window_size):
    """
    Compute the rolling population standard deviation.
    """
    # Write code here
    res = []
    n = len(values)

    for i in range(n - window_size + 1):
        res.append(np.std(values[i : i + window_size]))
    return res