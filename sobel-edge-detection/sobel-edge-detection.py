import numpy as np

kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
def find_gx(arr):
    k_x = kernel
    return np.sum(k_x * arr)
def find_gy(arr):
    k_y = kernel.T
    return np.sum(k_y * arr)
    
def sobel_edges(image):
    """
    Apply the Sobel operator to detect edges.
    """
    # Write code here
    image = np.array(image)

    h, w = image.shape
    
    new_h = (h - 3 + 1 * 2) + 1
    new_w = (w - 3 + 1 * 2) + 1

    image = np.pad(image,((1, 1), (1, 1)))
    
    res = np.zeros((new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):

            arr = image[i : i + 3, j: j + 3]
            
            gx = find_gx(arr)
            gy = find_gy(arr)

            res[i, j] = np.sqrt(gx**2 + gy**2)

    return res.tolist()
            
    
    