import numpy as np

def local_response_normalization(x: np.ndarray, k: float = 2, n: int = 5,
                                  alpha: float = 1e-4, beta: float = 0.75) -> np.ndarray:
    """
    Apply Local Response Normalization across channels.
    """
    # YOUR CODE HERE
    # pass
    a = np.array(x).copy()

    B, H, W, C = x.shape

    for b in range(B):
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    start = max(0, c - n//2)
                    end = min(C - 1, c + n // 2)
                    s = np.sum(x[b, h, w, start : end + 1]**2)
                    deno = (k + alpha * s) ** beta 
                    a[b, h, w, c] = x[b, h, w, c] / deno 
    return a