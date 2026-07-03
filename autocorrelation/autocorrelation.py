import numpy as np
def autocorrelation(series, max_lag):
    """
    Compute the autocorrelation of a time series for lags 0 to max_lag.
    """
    # Write code here
    n = len(series)
    mean = np.mean(series)
    gamma = np.sum((series - mean)**2)
    res = []
    k = max_lag
    for k in range(max_lag + 1):
        rk = 0
        for i in range(n - k):
            # res.append(float(/gamma))
            rk += (series[i] - mean) * (series[i + k] - mean)
        if gamma == 0:
            if k == 0:
                rk = 1.0
            else:
                rk = 0.0
        else:
            rk /= gamma
        res.append(rk)
        
    return res

    