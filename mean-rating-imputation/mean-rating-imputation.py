import numpy as np

def mean_rating_imputation(ratings_matrix, mode):
    ratings_matrix = np.asarray(ratings_matrix, dtype=float)

    # Đổi missing value thành nan
    temp = ratings_matrix.copy()
    temp[temp == 0] = np.nan

    if mode == "user":
        means = np.nanmean(temp, axis=1)      # mean mỗi hàng
        result = np.where(ratings_matrix == 0,
                          means[:, None],
                          ratings_matrix)
    else:
        means = np.nanmean(temp, axis=0)      # mean mỗi cột
        result = np.where(ratings_matrix == 0,
                          means[None, :],
                          ratings_matrix)
    result = np.where(np.isnan(result), 0, result)
    return result.tolist()