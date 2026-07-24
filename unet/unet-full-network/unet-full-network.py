import numpy as np

def unet(x: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """
    Complete U-Net: trace shape through 4 encoder blocks, bottleneck, 4 decoder blocks, output.
    Each block: two 3x3 unpadded convs (reduce by 4), encoder pools (halve), decoder upsamples (double).
    Returns zero array with correct output shape.
    """
    # Your implementation here
    N, H, W, C = x.shape
    # Encoder
    for i in range(4):
        H = (H-4)//2
        W = (W-4)//2

    #Bottle neck
    H -= 4
    W -= 4
    
    for i in range(4):
        H = H * 2 - 4
        W = W * 2 - 4
    return np.zeros((N, H, W, num_classes))