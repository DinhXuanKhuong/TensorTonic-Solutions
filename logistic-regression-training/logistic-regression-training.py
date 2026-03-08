import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.array(X)
    y = np.array(y)

    m, n = X.shape
    W = np.random.randn(n)

    b = 0.0

    print(X.shape)
    
    for step in range(steps):
        Z = X @ W + b 
        
        p = _sigmoid(Z)
        
        dW = X.T @ (p - y)
        dW /= m 
        db = np.sum(p - y) / m

        W -= lr * dW 
        b -= lr * db 
    return W, b