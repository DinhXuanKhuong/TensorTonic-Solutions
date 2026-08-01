import math 
def novelty_score(recommendations, item_counts, n_users):
    """
    Compute the average novelty of a recommendation list.
    """
    # Write code here
    novel = [-math.log2(item_counts[i]/n_users) for i in recommendations]
    res = 0.0 if len(novel) == 0 else sum(novel) / len(recommendations)
    return res