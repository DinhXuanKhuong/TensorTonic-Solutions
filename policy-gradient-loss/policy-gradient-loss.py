def policy_gradient_loss(log_probs, rewards, gamma):
    """
    Compute REINFORCE policy gradient loss with mean-return baseline.
    """
    # Write code here
    T = len(log_probs)
    G = [0 for i in range(T)]
    G[T - 1] = rewards[T - 1]
    for t in range(T - 2, -1, -1):
        G[t] = rewards[t] + gamma * G[t + 1]

    G_mean = sum(G) / T 

    A = [G[t] - G_mean for t in range(T)]

    L = -1 / T * sum([log_probs[t] * A[t] for t in range(T)])
    return L