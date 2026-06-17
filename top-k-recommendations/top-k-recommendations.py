import numpy as np
def top_k_recommendations(scores, rated_indices, k):
    """
    Return indices of top-k unrated items by predicted score.
    """
    # Write code here
    scores = np.asarray(scores, dtype = np.float64)
    # [5,1,3] -> [-inf, 1, 3] -> [inf, -1, -3] -> [1, 2, 0 ] ->  
    if (len(rated_indices)):
        print("hello")
        scores[list(rated_indices)] = -np.inf
        
    # print(scores)
    idx = np.argsort(-scores).tolist()
    

    k = min(k, len(scores) - len(rated_indices))

    return idx[:k]