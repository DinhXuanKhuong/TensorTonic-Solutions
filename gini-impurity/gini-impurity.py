import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    # Write code here

    if(len(y_left) + len(y_right) == 0):
        return 0.0
    left_vals, left_counts = np.unique(y_left, return_counts=True)

    # print(left_counts)
    
    gini_left = 0 if sum(left_counts) == 0 else 1 - np.sum(((left_counts / sum(left_counts))**2)) 

    
    right_vals, right_counts = np.unique(y_right, return_counts=True)

    # print(right_counts)
    gini_right = 0 if sum(right_counts) == 0 else 1 - np.sum(((right_counts / sum(right_counts))**2))

    # print(gini_right)
    n_left = len(y_left)
    n_right = len(y_right)
    n = n_left + n_right

    res = (n_left / n) * gini_left + (n_right / n) * gini_right
    # print(res)
    return res
    
    