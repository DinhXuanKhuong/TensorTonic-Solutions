import numpy as np

def func(c, x):
    k = np.arange(len(c))
    return np.sum(c * x**k)
def finite_difference_derivative(coefficients, x, h):
    """
    Returns: the polynomial value at x, the value at x plus h, and the forward-difference slope
    """
    fx = func(coefficients, x)
    fxh = func(coefficients, x + h)
    return fx, fxh, (fxh - fx)/ h
