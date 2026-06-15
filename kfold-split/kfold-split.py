import numpy as np

def kfold_split(N, k, shuffle=True, rng=None):
    """
    Returns: list of length k with tuples (train_idx, val_idx)
    """
    # Write code here
    # pass
    res = []
    indices = np.arange(N)
    if shuffle:
        if rng is not None:
            indices = rng.permutation(indices)
        else:
            indices = np.random.shuffle(indices)

    folds = np.array_split(indices, k)
    # print(folds)
    for fold_idx in range(k):
        val = folds[fold_idx]
        train = np.concatenate([folds[i] for i in range(k) if i != fold_idx])
        res.append((train, val))

    return res

    