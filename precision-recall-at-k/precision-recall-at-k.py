def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    top_k = recommended[:k]
    top_k_relevant = len(set(top_k).intersection(set(relevant))) 
    precisionK =  top_k_relevant / k   

    recallK = top_k_relevant / len(relevant)

    return [precisionK, recallK]