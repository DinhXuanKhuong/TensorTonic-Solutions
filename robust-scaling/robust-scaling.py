import numpy as np
def find_median(arr):
    
    n = len(arr)
    if n % 2 == 1:
        return arr[n//2]
    else:
        return (arr[n//2] + arr[n // 2 - 1]) / 2
def robust_scaling(values):
    """
    Scale values using median and interquartile range.
    """
    arr = values.copy()
    values.sort()
    # Write code here
    n = len(values)
    values = np.asarray(values)    
    if n <= 1:
        return [0.]

    median = find_median(values)
    q1 = find_median(values[:n//2])
    q3 = find_median(values[n//2:]) if n%2 == 0 else find_median(values[n//2+1:])
    x_scaled = np.where(arr == median, 0., (arr - median)/(q3 - q1)) 
    
    return x_scaled