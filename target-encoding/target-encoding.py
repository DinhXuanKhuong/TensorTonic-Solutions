def target_encoding(categories, targets):
    """
    Replace each category with the mean target value for that category.
    """
    # Write code here
    cate_val = dict()

    for i in range(len(categories)):
        cate_val[categories[i]] = cate_val.get(categories[i], [])
        cate_val[categories[i]].append(targets[i])

    res = [sum(cate_val[u])/len(cate_val[u]) for u in categories]

    return res