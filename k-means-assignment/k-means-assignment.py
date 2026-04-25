import numpy as np
def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    res = np.zeros(len(points))
    points = np.array(points)
    centroids = np.array(centroids)
    i = 0
    for p in points:
        indice = np.argmin(np.linalg.norm(centroids - p, axis = 1))
        res[i] = indice
        i +=1
    return res.tolist()