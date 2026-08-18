import numpy as np

def classification_head(encoder_output: np.ndarray, num_classes: int, W_head: np.ndarray = None) -> np.ndarray:
    """
    Classification head for ViT. Extract [CLS], LayerNorm, linear projection.
    W_head: projection matrix (D, num_classes). If None, initialize randomly.
    """
    # YOUR CODE HERE
    encoder_output = np.asarray(encoder_output)
    h_cls = encoder_output[:, 0, :]
    D = h_cls.shape[-1]
    if W_head is None:
        W_head = np.random.rand(D, num_classes)

    h = (h_cls - np.mean(h_cls, axis = -1, keepdims=True)) / (np.std(h_cls, axis = -1, keepdims=True) + 1e-8)

    logits = h @ W_head

    return logits