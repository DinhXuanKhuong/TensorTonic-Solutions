def popularity_ranking(items, min_votes, global_mean):
    """
    Compute the Bayesian weighted rating for each item.
    """
    # Write code here
    res = []
    for (r, v) in items:
        res.append(v / (v + min_votes) * r + min_votes / (v + min_votes) * global_mean)
    return res