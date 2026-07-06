import math 
def cyclic_encoding(values, period):
    """
    Encode cyclic features as sin/cos pairs.
    """
    # Write code here
    res = []

    for v in values:
        angle = 2 * math.pi * v / period
        res.append([math.sin(angle), math.cos(angle)])
    return res