def histogram_equalize(image):
    """
    Apply histogram equalization to enhance image contrast.
    """
    # Write code here
    freq = dict()
    h = len(image)
    w = len(image[0])

    for r in image:
        for v in r:
            freq[v] = freq.get(v, 0) + 1
    # freq_map = freq.copy()
    # for k in freq:
    #     freq[k] /= (h * w)
    freq = dict(sorted(freq.items()))
    keys = list(freq.keys())
    
    for i in range(1, len(keys)):
        freq[keys[i]] += freq[keys[i - 1]]
    cdf_min = min(freq.values())
    res = image.copy()
    
    for i in range(h):
        for j in range(w):
            res[i][j] = 0 if cdf_min == h * w else round((freq[image[i][j]] - cdf_min)/ (h * w - cdf_min) * 255)
    return res