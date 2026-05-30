import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    # Write code here
    # pass
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    if (y_true == y_true[0]).all():
        if (y_pred == y_true).all():
            return 1.
        else:
            return 0.
        
    deni = np.sum((y_true - np.mean(y_true))**2)
    
    r_sqr = 1 - np.sum((y_true - y_pred)**2) / deni

    return r_sqr