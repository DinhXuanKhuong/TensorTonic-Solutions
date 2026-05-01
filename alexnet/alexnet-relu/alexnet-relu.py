import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU activation: f(x) = max(0, x)
    """
    # YOUR CODE HERE
    # pass
    x = np.array(x)

    res = np.where(x > 0 , x, 0)

    return res