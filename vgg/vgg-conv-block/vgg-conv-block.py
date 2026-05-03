import numpy as np

def vgg_conv_block(x: np.ndarray, weights: list, biases: list) -> np.ndarray:
    """
    Returns: np.ndarray of shape (B, H, W, C_out) after sequential linear transforms with ReLU
    """
    # Your implementation here
    # weights = np.array(weights)
    # biases = np.array(biases)

    for i in range(len(weights)):
        x = x @ np.array(weights[i]) + np.array(biases[i])
        x = np.maximum(0, x)

    return x