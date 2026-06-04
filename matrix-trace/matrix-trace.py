import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    # Write code here
    # pass
    A = np.asarray(A)
    tr = 0
    for i in range(len(A)):
        tr += A[i][i]
    return tr
