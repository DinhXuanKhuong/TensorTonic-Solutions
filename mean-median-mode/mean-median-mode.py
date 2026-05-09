import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    # Write code here
    # pass
    x = np.array(x)
    m = np.mean(x)
    med = np.median(x)
    d = Counter(x)
    mode = -1
    mx = 0
    for k in d:
        if d[k] > mx:
            mx = d[k]
            mode = k 

    return m, med, mode