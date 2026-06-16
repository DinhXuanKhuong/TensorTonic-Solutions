import math
def bilinear_resize(image, new_h, new_w):
    """
    Resize a 2D grid using bilinear interpolation.
    """
    # Write code here
    h = len(image)
    w = len(image[0])
    res = [[0 for i in range(new_w)] for j in range(new_h)]
    for i in range(new_h):
        for j in range(new_w):
            src_y = 0 if new_h == 1 else i * (h - 1) / (new_h - 1)
            src_x = 0 if new_w == 1 else j * (w - 1) / (new_w - 1)

            y0, x0 = math.floor(src_y), math.floor(src_x)
            dy, dx = src_y - y0, src_x - x0 
            y1, x1 = min(y0 + 1, h - 1), min (x0 + 1, w - 1)

            res[i][j] = image[y0][x0] * (1 - dy) * (1 - dx) + image[y1][x0] * dy * (1 - dx) + image[y0][x1] * (1 - dy) * dx + image[y1][x1] * dy * dx 
    return res
            
    