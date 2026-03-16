import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    # pass
    y = np.array(y)
    if (len(y) == 0):
        return 0.0

    unq, cnt = np.unique(y, return_counts = True)

    cnt = np.array(cnt)
    s = np.sum(cnt)
    cnt = cnt / s

    lg2 = np.log2(cnt)

    h = -1 * np.sum(lg2 * cnt)
    return h
    

    