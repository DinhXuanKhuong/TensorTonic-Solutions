import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    # Write code here
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    
    e = y_true - y_pred

    loss = np.where(abs(e) <= delta, 0.5 * e**2, delta * (abs(e) - 0.5 * delta))

    return np.mean(loss)