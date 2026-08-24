import numpy as np

def mc_policy_evaluation(episodes: list, gamma: float, n_states: int) -> np.ndarray:
    """Return first-visit Monte Carlo state values."""

    returns = [[] for _ in range(n_states)]

    for episode in episodes:
        G = 0.0

        # Tính return G_t cho từng timestep
        episode_returns = [0.0] * len(episode)

        for t in range(len(episode) - 1, -1, -1):
            state, reward = episode[t]
            G = reward + gamma * G
            episode_returns[t] = G

        # First-visit MC
        visited = set()

        for t, (state, reward) in enumerate(episode):
            if state not in visited:
                returns[state].append(episode_returns[t])
                visited.add(state)

    # Trung bình return của mỗi state
    V = np.zeros(n_states)

    for state in range(n_states):
        if returns[state]:
            V[state] = np.mean(returns[state])

    return np.round(V, 4)