import numpy as np
def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    # Write code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = (y_pred == y_true)
    
    tp = np.sum(mask)
    fp_fn = len(mask) - tp 
    
    return 2*tp / (2 * tp + 2 * fp_fn)