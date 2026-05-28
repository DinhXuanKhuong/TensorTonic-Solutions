def remove_stopwords(tokens, stopwords):
    """
    Returns: list[str] - tokens with stopwords removed (preserve order)
    """
    # Your code here
    res = []
    for t in tokens:
        if t not in stopwords:
            res.append(t)
    return res