def double_exponential_smoothing(series, alpha, beta):
    """
    Apply Holt's linear trend method and return the level values.
    """
    # Write code here
    res = []
    l = series[0]
    t = series[1] - series[0]
    
    res.append(l)

    for i in range(1, len(series)):
        l2 = alpha * series[i]  + (1 - alpha) * (l + t)
        t = beta* (l2 - l) + (1 - beta) * t
        l = l2 
        res.append(l)
    return res