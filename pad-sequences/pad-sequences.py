import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    # seqs = np.array(seqs)

    if max_len is None:
        max_len = max([len(x) for x in seqs])
    res = []
    for i in range(len(seqs)):
        seq = np.array(seqs[i])
        pad = max(0, max_len - len(seq))
        
        print(pad)
        add = np.pad(seq, (0, pad), constant_values = pad_value)[:max_len]
        res.append(add)

    
    return np.array(res)
    # return seqs[:][:max_len]