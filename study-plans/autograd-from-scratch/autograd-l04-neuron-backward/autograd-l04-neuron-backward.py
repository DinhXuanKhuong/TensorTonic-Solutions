import torch

def neuron_backward(inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, upstream_gradient: torch.Tensor) -> tuple:
    """
    Returns a tuple of tensors: output, input gradients, weight gradients, bias gradient.
    """
    a = inputs @ weights + bias
    y = torch.tanh(a)

    sigma = upstream_gradient * (1 - y**2)
    input_grad = sigma * weights
    weight_grad = sigma * inputs

    bias_grad = sigma

    return y, input_grad, weight_grad, bias_grad
