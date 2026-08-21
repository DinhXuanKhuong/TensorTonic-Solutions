import numpy as np

def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    # Write code here
    rater1 = np.asarray(rater1)
    rater2 = np.asarray(rater2)
    n = rater1.shape[0]
    
    p_o = np.sum(rater1 == rater2) / n
    
    k = max(max(rater1), max(rater2)) + 1
    
    p_e = 0
    
    for i in range(k):
        p_e += np.sum(rater1 == i) * np.sum(rater2 == i) / (n**2)

    
    return 1.0 if (1 - p_e == 0) else (p_o - p_e) / (1 - p_e)