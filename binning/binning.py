def binning(values, num_bins):
    """
    Assign each value to an equal-width bin.
    """
    # Write code here
    n = len(values)
    w = (max(values) - min(values))/num_bins
    res = [0 for i in range(n)]
    if w == 0:
        return res
    for i in range(n):
        res[i] = min((values[i] - min(values)) // w, num_bins - 1)
    return res
    