def gae(rewards, values, gamma, lam):
    """
    Compute Generalized Advantage Estimation.
    """
    # Write code here
    td_error = []
    for t in range(len(values) - 1):
        delta_t = rewards[t] + gamma * values[t + 1] - values[t]
        td_error.append(delta_t)

    T = len(td_error)
    A = [0] * T
    A[T - 1] = td_error[T - 1]
    for t in range(T - 2, -1, -1):
        A[t] = td_error[t] + gamma * lam * A[t + 1]
    return A