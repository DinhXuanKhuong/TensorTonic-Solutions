import numpy as np

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def get_prob(x, D_W):
    x = np.asarray(x)
    D_W = np.asarray(D_W)
    
    return _sigmoid(x @ D_W)

def train_gan_step(real_data, fake_data, D_W):
    """
    Returns: dict with "d_loss" and "g_loss" as float values
    """
    # Your implementation here
    # pass
    epsilon = 1e-8
    p_real = np.clip(get_prob(real_data, D_W), epsilon, 1 - epsilon)
    p_fake = np.clip(get_prob(fake_data, D_W), epsilon, 1 - epsilon)

    d_loss = -np.mean(np.log(p_real) + np.log(1 - p_fake))

    g_loss = -np.mean(np.log(p_fake))

    return {"d_loss" : np.round(d_loss, 4),
           "g_loss": np.round(g_loss, 4)}