def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """
    # Write code here
    res = 0
    k = min(len(recommendations[0]), k)
    for i,u in enumerate(recommendations):
        res += (len(set(u[:k]) & set(ground_truth[i])) >= 1)
    return res / len(recommendations)
    