import numpy as np 
def priority_replay_sample(priorities, alpha, beta):
    """
    Compute sampling probabilities and importance sampling weights for PER.
    """
    # Write code here
    N = len(priorities)
    priorities = np.asarray(priorities)

    priorities = priorities**alpha

    P = priorities / np.sum(priorities)

    w = (N * P)**(-beta)

    w /= np.max(w)

    return [P.tolist(), w.tolist()]