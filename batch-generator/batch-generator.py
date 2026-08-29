import numpy as np

def batch_generator(X, y, batch_size: int, seed: int = 42, drop_last: bool = False):
    """
    Shuffles X and y using a single seed, then yields consecutive mini-batches.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    
    n_samples = len(X_arr)
    rng = np.random.default_rng(seed)
    
    # Generate a single index permutation and apply to both arrays
    indices = rng.permutation(n_samples)
    X_shuffled = X_arr[indices]
    y_shuffled = y_arr[indices]
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = start_idx + batch_size
        
        if drop_last and end_idx > n_samples:
            break
            
        yield X_shuffled[start_idx:end_idx], y_shuffled[start_idx:end_idx]