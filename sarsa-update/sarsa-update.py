def sarsa_update(q_table, state, action, reward, next_state, next_action, alpha, gamma):
    """
    Perform one SARSA update and return the updated Q-table.
    """
    # Write code here
    q = q_table.copy()
    td = reward + gamma * q[next_state][next_action] - q[state][action]
    q[state][action] += alpha * td 
    return q
    