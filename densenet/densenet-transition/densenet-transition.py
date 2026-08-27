import torch
import torch.nn.functional as F

def transition_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, out_channels, H//2, W//2) after BN-ReLU-1x1Conv then 2x2 average pooling.
    """
    # YOUR CODE HERE
    x = torch.tensor(x, dtype = torch.float64)
    C = x.shape[1]
    bn_beta = torch.tensor(bn_beta, dtype = torch.float64).view(1, C, 1, 1)
    bn_gamma = torch.tensor(bn_gamma, dtype = torch.float64).view(1, C, 1, 1)
    bn_mean = torch.tensor(bn_mean, dtype = torch.float64).view(1, C, 1, 1)
    bn_var = torch.tensor(bn_var, dtype = torch.float64).view(1, C, 1, 1)

    conv_weight = torch.tensor(conv_weight, dtype = torch.float64)

    x_norm = bn_gamma * (x - bn_mean) / (torch.sqrt(bn_var + eps)) + bn_beta

    x_norm = torch.relu(x_norm)

    y = F.conv2d(x_norm, conv_weight, stride = 1, bias = None)

    res = F.avg_pool2d(y, kernel_size = 2, stride = 2)
    return res
    
    
