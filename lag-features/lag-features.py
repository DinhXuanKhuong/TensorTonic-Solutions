def lag_features(series, lags):
    """
    Create a lag feature matrix from the time series.
    """
    # Write code here
    max_lag = max(lags)

    res = []
    for i in range(max_lag, len(series)):
        res.append([series[i - lag] for lag in lags])
    return res