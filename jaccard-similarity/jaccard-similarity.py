def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    # Write code here
    set_a = set(set_a)
    set_b = set(set_b)
    a = len(set_a & set_b)
    b = len(set_a | set_b)
    res = 0.0 if b == 0 else a / b
    return res