import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    """
    # Write code here
    total = np.sum(C)
    a = np.sum(C, axis = 0, keepdims = True) # 1, C
    b = np.sum(C, axis = 1, keepdims = True) # C, 1
    E = (b @ a / total)

    chi = np.sum((C - E)**2 / E)

    return chi, E
    
    