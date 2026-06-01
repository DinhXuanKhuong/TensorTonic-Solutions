def simple_moving_average(values, window_size):
    """
    Compute the simple moving average of the given values.
    """
    # Write code here
    res = []
    n = len(values)
    for i in range(1, n):
        values[i] = values[i] + values[i - 1]
    # 10 30 60 100
    values.append(0)
    
    for i in range(0, n - window_size + 1):
        res.append((values[i + window_size - 1] - values[i - 1])/window_size)
    return res
        
        