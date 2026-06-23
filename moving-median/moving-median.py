def find_median(arr):
    n = len(arr)
    arr.sort()
    if n % 2 == 1:
        return arr[n//2] * 1.0
    else:
        return (arr[n//2] + arr[n//2 - 1])/2
def moving_median(values, window_size):
    """
    Compute the rolling median for each window position.
    """
    # Write code here
    res = []
    
    for i in range(len(values) - window_size + 1):
        res.append(find_median(values[i : i + window_size]))
    return res