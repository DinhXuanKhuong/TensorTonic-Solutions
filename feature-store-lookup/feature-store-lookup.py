def feature_store_lookup(feature_store, requests, defaults):
    """
    Join offline user features with online request-time features.
    """
    # Write code here
    res = []
    for req in requests:
        user_id = req["user_id"]
        user_feature = feature_store.get(user_id, defaults)
        res.append(user_feature | req["online_features"])
    return res