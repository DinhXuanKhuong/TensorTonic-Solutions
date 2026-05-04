def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    # Your code here
    res = dict()
    for s in sentences:
        for w in s:
            res[w] = res.get(w, 0) + 1
    return res