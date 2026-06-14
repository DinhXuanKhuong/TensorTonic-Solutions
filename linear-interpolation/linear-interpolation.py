def linear_interpolation(values):
    """
    Fill missing (None) values using linear interpolation.
    """
    # Write code here
    res = values.copy()
    n = len(res)
    r = n - 1
    l = 0
    
    for i in range(n):
        if values[i] is not None:
            continue
        for left in range(i, 0, -1):
            if res[left] is not None:
                l = left
                break 
        for right in range(i, n):
            if res[right] is not None:
                r = right
                break
        res[i] = res[l] + (i - l) / (r - l) * (res[r] - res[l])
    return res