def rank_transform(values):
    """
    Replace each value with its average rank.
    """
    # Write code here
    x = values.copy()
    x.sort()
    
    rank = dict()
    
    for i, val in enumerate(x):
        rank[val] = rank.get(val, []) + [i + 1]

    for i in range(len(values)):
        values[i] = sum(rank[values[i]]) / len(rank[values[i]])
    return values