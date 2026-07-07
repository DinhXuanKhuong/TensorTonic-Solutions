def percent_change(series):
    """
    Compute the fractional change between consecutive values.
    """
    # Write code here
    res = []
    for i in range(1, len(series)):
        pct = 0. if series[i-1] == 0 else (series[i] - series[i-1]) / series[i-1]
        res.append(pct)
    return res