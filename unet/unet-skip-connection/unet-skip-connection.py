import numpy as np

def crop_and_concat(encoder_features: np.ndarray, decoder_features: np.ndarray) -> np.ndarray:
    """
    Crop encoder features to match decoder spatial dims, then concatenate along channels.
    """
    # Your implementation here
    _, H_e, W_e, _ = encoder_features.shape
    _, H_d, W_d, _ = decoder_features.shape

    start_H = (H_e - H_d)//2
    start_W = (W_e - W_d)//2

    new_encoder = encoder_features[:, start_H: start_H + H_d, start_W : start_W + W_d, :].copy()
    return np.concat([new_encoder, decoder_features], axis = -1)