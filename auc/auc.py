import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    # Write code here
    n = len(fpr)
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    
    a = tpr[:n - 1] + tpr[1:n]
    b = fpr[1:n] - fpr[:n - 1]

    auc = 0.5 * np.sum(a * b)
    return auc