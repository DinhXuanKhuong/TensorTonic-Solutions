import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    # Your code here
    mean_x = np.mean(x, axis = -1, keepdims = True)
    var_x = np.var(x, axis = -1, keepdims = True)
    res = (x - mean_x) / np.sqrt(var_x + eps)

    return gamma * res + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    # Your code here
    batch, seq_len, d_model = Q.shape
    
    # print(batch, seq_len, d_model)
    
    Q = Q @ W_q
    K = K @ W_k
    V = V @ W_v

    d_head = d_model //  num_heads
    # print(d_head)

    keys = K.reshape(batch, seq_len, num_heads, d_head)
    queries = Q.reshape(batch, seq_len, num_heads, d_head)
    values = V.reshape(batch, seq_len, num_heads, d_head)

    keys = keys.transpose(0, 2, 1, 3) #(b, n_head, seq_len, d_head)
    queries = queries.transpose(0, 2, 1, 3) #(b, n_head, seq_len, d_head)
    values = values.transpose(0, 2, 1, 3) #(b, n_head, seq_len, d_head)

    attn_scores = ((queries @ keys.transpose(0, 1, 3, 2)) / keys.shape[-1]**0.5)
    attn_scores = softmax(attn_scores)

    context_vec = attn_scores @ values

    context_vec = context_vec.transpose(0, 2, 1, 3)

    context_vec = context_vec.reshape(batch, seq_len, -1)


    mul_attn = context_vec @ W_o

    return mul_attn

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    # Your code here
    hidden = np.maximum(0, x @ W1 + b1)
    
    output = hidden @ W2 + b2 
    return output

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # Your code here
    pass

    mha = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    lnorm_1 = layer_norm(x + mha, gamma1, beta1)
    ffw = feed_forward(lnorm_1, W1, b1, W2, b2)
    lnorm_2 = layer_norm(lnorm_1 + ffw, gamma2, beta2)

    return lnorm_2