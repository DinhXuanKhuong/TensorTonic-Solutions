import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    # Write code here
    N = len(documents)
    vocab = set()
    arr = []
    doc_counter = []
    for doc in documents:
        tmp = doc.lower().split(" ")
        arr.append(tmp)
        vocab.update(tmp)
        doc_counter.append(Counter(tmp))

    vocab = list(vocab)
    # print(vocab)
    vocab.sort()
    
    k = len(vocab)
    
    res = np.zeros((N,k))

    for i in range(N):
        for j in range(k):
            tf = doc_counter[i][vocab[j]] / len(arr[i])
            cnt = 0
            for d in doc_counter:
                cnt += (d[vocab[j]] != 0)
            idf = math.log(N / cnt)
            res[i][j] = tf * idf
    return res, vocab