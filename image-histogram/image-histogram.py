def image_histogram(image):
    """
    Compute the intensity histogram of a grayscale image.
    """
    # Write code here
    bins = [0] * 256
    h = len(image)
    w = len(image[0])
    for i in range(h):
        for j in range(w):
            bins[image[i][j]] += 1
    return bins