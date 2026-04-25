import numpy as np


def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    # Write code here
    points = np.array(points)
    assignments = np.array(assignments)
    # k = np.scalar(k)
    res = []
    for c in range(k):
        p_indices_list = np.where(assignments == c)
        p_list = points[list(p_indices_list), :][0]
        if (len(p_list) != 0):
            centroid = np.mean(p_list, axis = 0)
        else:
            centroid = np.zeros(len(points[0]))
            
        res.append(centroid.tolist())
    return res