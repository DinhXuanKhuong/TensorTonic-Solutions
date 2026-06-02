import numpy as np
from collections import Counter
def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here
    d = Counter(tokens)
    res = []
    for word in vocab:
        cnt = d.get(word, 0)
        res.append(cnt)
    return np.array(res).astype(int)