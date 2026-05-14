import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITH skip connections.
    Gradient at layer l = sum of paths through network
    """
    # YOUR CODE HERE
    new_x = x.copy()
    for M in gradients_F:
        M = np.array(M) + np.eye(len(new_x))
        new_x = new_x @ M  

    return new_x

def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITHOUT skip connections.
    """
    # YOUR CODE HERE
    new_x = x.copy()
    for M in gradients_F:
        M = np.array(M)
        new_x = new_x @ M
    return new_x
    
