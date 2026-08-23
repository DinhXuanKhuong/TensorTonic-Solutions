import numpy as np

def compute_advantage(states, rewards, V, gamma):
    """
    Returns: A (NumPy array of advantages)
    """
    # Write code here
    n = len(rewards)
    states = np.asarray(states)
    rewards = np.asarray(rewards)
    V = np.asarray(V)
    g = np.zeros(n + 1, dtype = float)
    
    for i in range(n - 1, -1, -1):
        g[i] = rewards[i] + gamma * g[i + 1]
    
    A = np.round(g[:n] - V[states], 4)
    return A