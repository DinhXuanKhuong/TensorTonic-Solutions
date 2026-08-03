import numpy as np
def replay_buffer_sample(buffer, batch_size, seed):
    """
    Sample a batch of transitions from the replay buffer.
    """
    # Write code here
    buffer = np.asarray(buffer)
    
    rng = np.random.RandomState(seed=seed)
    
    indices = rng.choice(len(buffer), size=batch_size, replace = False)
    print(indices)
    return buffer[indices, :]