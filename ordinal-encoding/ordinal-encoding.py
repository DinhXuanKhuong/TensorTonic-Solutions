def ordinal_encoding(values, ordering):
    """
    Encode categorical values using the provided ordering.
    """
    # Write code here
    rank = dict()

    for i, val in enumerate(ordering):
        rank[val] = i 

    res = [rank[val] for val in values]

    return res