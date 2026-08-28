import math

def rotate_image(image: list, angle_degrees: float) -> list:
    """
    Returns the counterclockwise nearest-neighbor rotation.
    """
    # Write code here
    H = len(image)
    W = len(image[0])
    c_y = (H - 1) / 2
    c_x = (W - 1) / 2
    
    s = set()
    for r in image:
        for v in r:
            s.add(v)
    
    theta = angle_degrees * math.pi / 180
    
    res = [[0 for j in range(W)] for i in range(H)]
    for i in range(H):
        for j in range(W):
            dy = i - c_y
            dx = j - c_x
            s_y = int(round(c_y + dy * math.cos(theta) + dx * math.sin(theta)))
            s_x = int(round(c_x - dy * math.sin(theta) + dx * math.cos(theta)))
            if (s_x >= 0 and s_y >= 0 and s_x < H and s_y < W) and image[s_x][s_y] in s:
                res[i][j] = image[s_y][s_x]
    return res
                
                
    