def discount_returns(rewards, gamma):
    """
    Compute the discounted return at every timestep.
    """
    # Write code here
    T = len(rewards)
    res = [0 for i in range(T)]
    res[T-1] = rewards[T-1]
    for i in range(T-2, -1, -1):
        res[i] = rewards[i] + gamma * res[i+1]
    return res