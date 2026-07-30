def adjusted_cosine_similarity(ratings_matrix, item_i, item_j):
    """
    Compute adjusted cosine similarity between two items.
    """
    # Write code here
    U = len(ratings_matrix)
    I = len(ratings_matrix[0])
    mu = []
    for u in ratings_matrix:
        n_zeros = 0
        s = 0
        for i in u:
            n_zeros += (i == 0)
            s += i 
        mu.append(s / (I - n_zeros))
        print(s / (I - n_zeros))
    a = sum([(ratings_matrix[u][item_i] - mu[u]) * (ratings_matrix[u][item_j] - mu[u]) if (ratings_matrix[u][item_i] != 0) and (ratings_matrix[u][item_j] !=0) else 0 for u in range(U)] )
    b1 = (sum([(ratings_matrix[u][item_i] - mu[u])**2  if (ratings_matrix[u][item_i] != 0) and (ratings_matrix[u][item_j] !=0) else 0  for u in range(U)]))**0.5
    b2 = (sum([(ratings_matrix[u][item_j] - mu[u])**2 if (ratings_matrix[u][item_i] != 0) and (ratings_matrix[u][item_j] !=0) else 0 for u in range(U)]))**0.5
    b = b1 * b2 
    res = 0.0 if b == 0 else a / b
    return res