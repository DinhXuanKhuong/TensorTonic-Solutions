import numpy as np

def detect_skew(train_dist, serving_dist, threshold=0.2, eps=1e-10):
    """
    Detect train-serving skew using PSI.
    """
    # Write code here
    res = dict()
    for key in train_dist:
        train = np.asarray(train_dist[key], dtype=np.float64) + eps
        serving = np.asarray(serving_dist[key], dtype=np.float64) + eps
        psi = np.sum((serving - train) * np.log(serving / train))
        skewed = (psi >= threshold)

        res[key] = {"psi": float(psi), "skewed" : bool(skewed)}
    return res