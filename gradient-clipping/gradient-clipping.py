import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    # Write code here
    g = np.array(g)

    
    g_norm = np.linalg.norm(g)
    
    if g_norm == 0:
        return np.zeros_like(g)

    if max_norm <= 0:
        return g
    # print(g_norm)

    print(g_norm)
    
    g_clipped = g if g_norm <= max_norm else g * max_norm / g_norm 

    print(g * max_norm / g_norm)

    return g_clipped