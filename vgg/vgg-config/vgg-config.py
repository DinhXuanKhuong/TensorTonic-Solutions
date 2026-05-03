import numpy as np

def make_vgg_config(variant: str) -> list:
    """
    Return the layer configuration for a VGG variant.
    """
    # Your implementation here
    config = {
        'vgg11': [1, 1, 2, 2, 2],
        'vgg13': [2, 2, 2, 2, 2],
        'vgg16': [2, 2, 3, 3, 3],
        'vgg19': [2, 2, 4, 4, 4]
    }

    res = []

    version = config[variant.lower()]
    
    for i, k in enumerate(version):
        x = i if i != 4 else 3
        res += [64 * (2 ** x) for h in range(k)]
        res.append('M')

    return res