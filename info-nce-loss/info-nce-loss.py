import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    # Write code here
    # pass
    Z1 = np.asarray(Z1)
    Z2 = np.asarray(Z2)
    N = Z1.shape[0]
    
    S = (Z1 @ Z2.T) / temperature
    S = S - np.max(S)
    
    e_S = np.exp(S)
    s_1 = np.sum(e_S, axis = 1)

    
    l = (-1/N) * np.sum(np.log(np.diag(e_S) / s_1)) 
    
    return l
    