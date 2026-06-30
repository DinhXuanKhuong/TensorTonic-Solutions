def weighted_moving_average(values, weights):
    """
    Compute the weighted moving average using the given weights.
    """
    # Write code here
    k = len(weights)
    n = len(values)
    res = []
    for i in range(n - k + 1):
        a = 0
        b = 0
        for j in range(k):
            a += weights[j] * values[i + j]
            b += weights[j]
        res.append(a / b)
    return res
            