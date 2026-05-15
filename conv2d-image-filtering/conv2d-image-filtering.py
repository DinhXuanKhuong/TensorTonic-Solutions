import numpy as np 

def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution to a single-channel image.
    """
    # Write code here
    image = np.array(image)
    kernel = np.array(kernel)

    h, w = image.shape

    h_f, w_f = kernel.shape

    h_out = (h + 2 * padding - h_f) // stride + 1
    w_out = (w + 2 * padding - w_f) // stride + 1

    res = np.zeros((h_out, w_out))

    image = np.pad(image, ((padding, padding), (padding, padding)))
                   
    for i in range(h_out):
        for j in range(w_out):
            arr = image[i * stride: i * stride + h_f, j * stride: j * stride + w_f]
            res[i, j] = np.sum(arr * kernel)

    return res.tolist()
    