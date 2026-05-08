import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    
    prob_pred = y_pred[np.arange(len(y_true)), y_true]

    CE = -1 * np.mean(np.log(prob_pred))

    return CE 

    
    