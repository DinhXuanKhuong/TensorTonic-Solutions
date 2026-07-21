import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    # YOUR CODE HERE
    # pass
    B, T, input_dim = X.shape
    
    h_prev = h_0.copy()
    hidden_states = np.zeros((B, T, h_0.shape[-1]))
    
    for t in range(T):
        h_t = np.tanh(X[:, t, :] @ W_xh.T + h_prev @ W_hh.T + b_h)
        hidden_states[:, t, :] = h_t
        h_prev = h_t

    return hidden_states, h_prev
        