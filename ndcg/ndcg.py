import math

def compute_dcg(arr, k):
    dcg = 0
    for i in range(1, k + 1):
        dcg += (2**arr[i-1] - 1) / (math.log2(i + 1))
    return dcg

def ndcg(relevance_scores, k):
    """
    Compute NDCG@k.
    """
    # Write code here
    k = min(len(relevance_scores), k)
    dcg = compute_dcg(relevance_scores, k)

    relevance_scores.sort(reverse = True)

    idcg = compute_dcg(relevance_scores, k)

    res = 0. if (idcg == 0) else dcg / idcg
    return res
    