def differencing(series, order):
    """
    Apply d-th order differencing to the time series.
    """
    # Write code here
    res = series.copy()
    for i in range(order):
        tmp = res.copy()
        for t in range(1, len(res)):
            res[t] = tmp[t] - tmp[t-1]
        res = res[1:].copy()
    return res