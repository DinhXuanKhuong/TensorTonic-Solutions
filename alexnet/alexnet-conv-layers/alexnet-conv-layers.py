import numpy as np

def alexnet_conv1(image: np.ndarray) -> np.ndarray:
    """
    AlexNet first conv layer: 11x11, stride 4, 96 filters (shape simulation).
    """
    # YOUR CODE HERE
    # pass
    b, h, w, c = image.shape
    h_out = (h + 2 * 2  - 11) // 4 + 1
    res = np.zeros((b, h_out, h_out, 96))
    return res