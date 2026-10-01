import torch

def muon_spectral_update(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    previous_momentum: torch.Tensor,
    momentum_coefficient: int | float,
    learning_rate: int | float,
) -> dict:
    """
    Returns a dict of tensors:
    new_parameter, new_momentum, orthogonalized_update.
    """

    new_momentum = (
        momentum_coefficient * previous_momentum
        + gradient
    )

    original_dtype = new_momentum.dtype

    # SVD does not support fp16/bf16 on CPU.
    if original_dtype in (torch.float16, torch.bfloat16):
        svd_input = new_momentum.float()
    else:
        svd_input = new_momentum

    U, _, Vt = torch.linalg.svd(
        svd_input,
        full_matrices=False
    )

    O = U @ Vt

    # Return to original dtype.
    O = O.to(original_dtype)

    new_parameter = parameter - learning_rate * O

    return {
        "new_parameter": new_parameter,
        "new_momentum": new_momentum,
        "orthogonalized_update": O,
    }