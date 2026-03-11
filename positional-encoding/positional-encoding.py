import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    pe = np.zeros((seq_len, d_model))
    pos = np.arange(seq_len)[:, np.newaxis]
    indice = np.arange(d_model)

    div_term = np.power(base, 2 * (indice // 2) / d_model)
    
    pe[:, 0::2] = np.sin(pos / div_term[0::2])
    pe[:, 1::2] = np.cos(pos / div_term[1::2])

    return pe