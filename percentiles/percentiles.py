import numpy as np

def cal_value(x, q):
    n = len(x)
    index = q * (n - 1)
    if(index.is_integer()):
        return x[index]
    else:
        i1 = int(index)
        i2 = i1 + 1 

        return x[i1] + (x[i2] - x[i1]) * (index - i1) 
    

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    # Write code here
    # pass
    q = np.array(q)
    
    x = np.sort(x)
    res = np.percentile(x, q, method = "linear")
    return res