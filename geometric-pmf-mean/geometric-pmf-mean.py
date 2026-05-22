import numpy as np

def geometric_pmf_mean(k, p):
    """
    Compute Geometric PMF and Mean.
    """
    # Write code here
    # pass
    p = np.array(p)
    k = np.array(k)
    
    E = 1 / p 
    
    pmf = np.power(1-p, k - 1) * p 
        
    # pmf = [((1 - p)**(num - 1) * p) for num in k]
    # E = 1 / p 
    return pmf, E