import numpy as np

def dis(a, b):
    return np.sum((a - b)**2, axis = -1)
    
def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    # Write code here
    anchor = np.asarray(anchor)
    positive = np.asarray(positive)
    negative = np.asarray(negative)
    print(dis(anchor, positive))
    print(dis(anchor, negative))
    

    L =np.mean(np.maximum(0, dis(anchor, positive) - dis(anchor, negative) + margin))
    return L
    
    