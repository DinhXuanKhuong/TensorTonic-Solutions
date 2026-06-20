import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # Write code here
    # pass
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    if y_train.shape[0] == 0:
        return []
    
    values, count = np.unique(y_train, return_counts = True)

    most_appearance = values[np.argmax(count)]

    return np.full(len(X_test), most_appearance)