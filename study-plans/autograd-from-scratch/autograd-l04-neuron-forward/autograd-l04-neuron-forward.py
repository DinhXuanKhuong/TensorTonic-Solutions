import torch

def neuron_forward(inputs, weights, bias):
    """
    Returns: scalar preactivation and tanh output
    """
    a = weights @ inputs + bias 
    y = torch.tanh(a)
    return a, y