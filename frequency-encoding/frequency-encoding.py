def frequency_encoding(values):
    """
    Replace each value with its frequency proportion.
    """
    # Write code here
    n = len(values)
    
    freq = dict()
    for ele in values:
        freq[ele] = freq.get(ele, 0) + 1
    
    res = [freq[u]/n for u in values]

    return res
    