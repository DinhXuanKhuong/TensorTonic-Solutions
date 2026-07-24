def cumulative_returns(returns):
    """
    Compute the cumulative return at each time step.
    """
    # Write code here
    n = len(returns)
    res = []
    for t in range(n):
        w_t = 1
        for i in range(t + 1):
            w_t *= (1 + returns[i])
        res.append(w_t - 1)
    return res