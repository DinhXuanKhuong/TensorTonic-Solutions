import numpy as np

def sinusoi(pos, i, d_model):
    k = i // 2
    if i % 2 == 0:
        return np.sin(pos / (10000**(2 * k / d_model)))
    else:
        return np.cos(pos / (10000**(2 * k / d_model)))

    

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here

    res = np.array([[sinusoi(pos, i, d_model) for i in range(d_model)] for pos in range(seq_length)])
    return res