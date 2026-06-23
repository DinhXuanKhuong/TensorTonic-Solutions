def user_based_cf_prediction(similarities, ratings):
    """
    Predict a rating using user-based collaborative filtering.
    """
    # Write code here
    nume = 0
    deno = 0
    for i in range(len(similarities)):
        nume += (similarities[i] * ratings[i]) * (similarities[i] > 0)
        deno += similarities[i] * (similarities[i] > 0)
    
    return 0.0 if deno == 0 else nume / deno  
    
    