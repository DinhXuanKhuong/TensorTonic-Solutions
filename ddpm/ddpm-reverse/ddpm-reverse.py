import numpy as np

def reverse_step(
    x_t: list,
    t: int,
    epsilon_pred: list,
    betas: list[float],
    z: list = None
) -> list:
    """
    Returns x at timestep t - 1, rounded to four decimals.
    Assumes timesteps are numbered 1, ..., T.
    """
    x_t = np.asarray(x_t, dtype=np.float64)
    betas = np.asarray(betas, dtype=np.float64)
    epsilon_pred = np.asarray(epsilon_pred, dtype=np.float64)

    beta_t = betas[t - 1]
    alpha_t = 1.0 - beta_t
    alpha_bar_t = np.cumprod(1.0 - betas)[t - 1]

    mu_t = (
        1.0 / np.sqrt(alpha_t)
        * (
            x_t
            - beta_t / np.sqrt(1.0 - alpha_bar_t)
            * epsilon_pred
        )
    )

    res = mu_t

    if t > 1 and z is not None:
        res = res + np.sqrt(beta_t) * np.asarray(z, dtype=np.float64)

    return np.round(res, 4)