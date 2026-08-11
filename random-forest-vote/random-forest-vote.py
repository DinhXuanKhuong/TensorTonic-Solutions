import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    # Write code here
    predictions = np.asarray(predictions)
    res = []
    
    for i in range(len(predictions[0])):
        sample = predictions[:, i]
        # print(sample)
        value, count = np.unique(sample, return_counts = True)
        res.append(value[np.argmax(count)])
    return res
    