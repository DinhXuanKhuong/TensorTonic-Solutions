import numpy as np
def log_transform(values):
    """
    Apply the log1p transformation to each value.
    """
    # Write code here
    values = np.array(values)
    values = values + 1
    return np.log(values)