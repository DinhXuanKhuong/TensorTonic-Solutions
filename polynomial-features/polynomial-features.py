def polynomial_features(values, degree):
    """
    Generate polynomial features for each value up to the given degree.
    """
    # Write code here
    res = [[num ** i for i in range(degree + 1)] for num in values]
    return res