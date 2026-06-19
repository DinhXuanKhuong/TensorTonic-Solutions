import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    # Write code here
    try:
        matrix = np.asarray(matrix)
        res = np.linalg.eigvals(matrix)
        np.lexsort(res)
        print(res)
        return res
    except Exception as e:
        print(e)
        return None
    # matrix = np.asarray(matrix)
    # try:
    #     res = np.linalg.eigvals(matrix)
    #     res = np.lexsort(res)
    # except np.linalg.LinAlgError:
    #     res = None
    # return res
    