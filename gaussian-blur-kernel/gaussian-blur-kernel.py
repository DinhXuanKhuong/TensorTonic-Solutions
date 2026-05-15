import math
def gaussian_kernel(size, sigma):
    """
    Generate a normalized 2D Gaussian blur kernel.
    """
    # Write code here

    kernel = [[0 for i in range(size)] for j in range(size)]

    center = size // 2
    s = 0
    for i in range(size):
        for j in range(size):
            x = j - center 
            y = i - center 

            kernel[i][j] = math.exp(-(x**2 + y**2) / (2 * sigma**2))

        s += sum(kernel[i])

    kernel = [[kernel[i][j] / s for j in range(size)] for i in range(size)]

    return kernel

    