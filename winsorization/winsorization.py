def ceil(x):
    return int(x)

def floor(x):
    r = x - (x // 1)
    return int(x) + (r > 0)

def find_k(p, n):
    return (n - 1) * p / 100

def winsorize(values, lower_pct, upper_pct):
    """
    Clip values at the given percentile bounds.
    """
    # Write code here
    v = values.copy()

    values.sort()
    
    n = len(values)
    k_lower = find_k(lower_pct, n)
    k_upper = find_k(upper_pct, n)

    v_lower = values[ceil(k_lower)] + (k_lower - ceil(k_lower)) * (values[floor(k_lower)] - values[ceil(k_lower)])

    v_upper = values[ceil(k_upper)] + (k_upper - ceil(k_upper)) * (values[floor(k_upper)] - values[ceil(k_upper)])

    for i in range(n):
        if v[i] < v_lower:
            v[i] = v_lower
        if v[i] > v_upper:
            v[i] = v_upper
    return v

    