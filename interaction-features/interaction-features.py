def find_interact(arr):
    n = len(arr)
    res = arr.copy()
    for i in range(n - 1):
        for j in range(i + 1, n):
            res.append(arr[i] * arr[j])
    return res
def interaction_features(X):
    """
    Generate pairwise interaction features and append them to the original features.
    """
    # Write code here
    res = []
    for arr in X:
        res.append(find_interact(arr))
    return res