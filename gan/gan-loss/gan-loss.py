import numpy as np

def discriminator_loss(real_probs, fake_probs):
    """Compute discriminator loss using binary cross-entropy.
    Returns: Loss value rounded to 4 decimals."""
    # pass
    real_probs = np.clip(np.asarray(real_probs), 1e-8, 1 - 1e-8)
    fake_probs = np.clip(np.asarray(fake_probs), 1e-8, 1 - 1e-8)
    

    log_real = np.log(real_probs)
    log_fake = np.log(1 - fake_probs)

    loss = -np.mean(log_real + log_fake)
    return loss 

def generator_loss(fake_probs):
    """Compute non-saturating generator loss.
    Returns: Loss value rounded to 4 decimals."""
    # pass
    fake_probs = np.clip(np.asarray(fake_probs), 1e-8, 1 - 1e-8)
    
    return -np.mean(np.log(fake_probs))