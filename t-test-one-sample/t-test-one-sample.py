import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    # Write code here
    x = np.asarray(x)
    n = len(x)
    mu = np.mean(x)
    
    s = (np.sum((x - mu)**2) / (n - 1))**0.5

    t = (mu - mu0)/(s / np.sqrt(n))
    return t