import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    Use the _entropy() helper above.
    """
    # Write code here
    # pass
    y = np.asarray(y)
    n = len(y)
    split_mask = np.asarray(split_mask)

    left_split = y[split_mask]
    right_split = y[~split_mask]

    IG = _entropy(y) - (len(left_split)/n * _entropy(left_split) + len(right_split)/n * _entropy(right_split))

    return IG
