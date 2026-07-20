import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class GRU:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        self.W_r = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_z = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_h = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_r = np.zeros(hidden_dim)
        self.b_z = np.zeros(hidden_dim)
        self.b_h = np.zeros(hidden_dim)

        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> tuple:
        """
        Forward pass. Returns (y, h_last).
        """
        # YOUR CODE HERE
        # pass
        N, T, input_dim = X.shape
        hidden_dim = self.W_r.shape[0]
        h_prev =  np.zeros((N, hidden_dim))
        print(T)
        y = np.zeros((N, T, self.W_y.shape[0]))
        
        for t in range(T):
            x_t = X[:, t, :]

            print(x_t.shape)

            rt = sigmoid( np.concat([h_prev, x_t], axis = -1) @ self.W_r.T + self.b_r)
            zt = sigmoid( np.concat([h_prev, x_t], axis = -1) @ self.W_z.T + self.b_z)
            h_ridle = np.tanh(np.concat([rt * h_prev, x_t], axis = -1) @ self.W_h.T + self.b_h)

            ht = zt * h_prev + (1 - zt) * h_ridle
            y[:, t, :] = ht @ self.W_y.T + self.b_y
            h_prev = ht

        # yt = ht @ self.W_y.T + self.b_y
        
        return y, ht
        