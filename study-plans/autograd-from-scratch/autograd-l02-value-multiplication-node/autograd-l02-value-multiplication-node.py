def value_multiplication_node(left: dict, right: dict, output_id: str) -> dict:
    """
    Returns the node dictionary with id, data, grad, op, and parents.
    """
    id_left = left['id']
    data_left = left['data']
    id_right = right['id']
    data_right = right['data']

    res = dict()
    res["id"] = output_id

    res["data"] = float(data_left * data_right)
    
    res["grad"] = 0.0

    res['op'] = '*'
    res["parents"] = [left, right]
    return res