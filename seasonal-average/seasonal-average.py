def seasonal_average(series, period):
    """
    Compute the average value for each position in the seasonal cycle.
    """
    # Write code here
    res = []
    for i in range(period):
        k = i
        s = 0
        cnt = 0
        while (k < len(series)):
            s += series[k]
            cnt += 1
            k += period
        res.append(s / cnt)
    return res