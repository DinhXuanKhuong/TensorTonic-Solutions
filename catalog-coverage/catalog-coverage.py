def catalog_coverage(recommendations, n_items):
    """
    Compute the catalog coverage of a recommender system.
    """
    # Write code here
    s = set()
    for re in recommendations:
        s = s | set(re)
    return 0 if n_items == 0 else len(s) / n_items