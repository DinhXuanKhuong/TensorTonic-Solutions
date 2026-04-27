import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code her e
    batch, seq_len, d_model = Q.shape
    
    print(batch, seq_len, d_model)
    
    Q = Q @ W_q
    K = K @ W_k
    V = V @ W_v

    d_head = d_model //  num_heads
    print(d_head)

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

    
        
        
    