import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute confusion matrix with optional normalization.
    """
    # Write code here
    # pass
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    
    n = len(y_pred)
    
    if num_classes is None:
        num_classes =  max(y_pred) + 1

    cm = np.zeros((num_classes, num_classes))

    for i in range(n):
        cm[y_true[i], y_pred[i]] += 1
            
    if normalize == 'true':
        cm = cm / (np.sum(cm, axis = 1, keepdims = True) + 1e-6)
    elif normalize == 'pred':
        cm = cm / (np.sum(cm, axis = 0, keepdims = True) + 1e-6)
    elif normalize == 'all':
        cm = cm / (np.sum(cm) + 1e-6)

    return cm
        