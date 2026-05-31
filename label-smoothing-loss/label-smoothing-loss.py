def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    # Write code here
    k = len(predictions)
    res = 0
    for i in range(k):
        q = epsilon / k + (i == target) * (1 - epsilon)
        res += q * math.log(predictions[i])
    return -res