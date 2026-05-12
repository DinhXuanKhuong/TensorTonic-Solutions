import numpy as np
def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    # Write code here
    cos_vec = np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

    L = 1 - cos_vec if label == 1 else max(0, cos_vec - margin)

    return L