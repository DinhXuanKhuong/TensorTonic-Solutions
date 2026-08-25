import torch
import torch.nn.functional as F

def composite_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, growth_rate, H, W): BN, ReLU, then a 3x3 same-padding convolution.
    """
    # YOUR CODE HERE
    x = torch.as_tensor(x, dtype = torch.float64)
    C = x.shape[1] 
    print(C)
    bn_mean = torch.as_tensor(bn_mean, dtype = torch.float64).reshape(1, C, 1, 1)
    bn_var = torch.as_tensor(bn_var, dtype = torch.float64).reshape(1, C, 1, 1)
    bn_gamma = torch.as_tensor(bn_gamma, dtype = torch.float64).reshape(1, C, 1, 1)
    bn_beta = torch.as_tensor(bn_beta, dtype = torch.float64).reshape(1, C, 1, 1)
    conv_weight = torch.as_tensor(conv_weight, dtype = torch.float64)
    print("NO")    
    x_norm = (x - bn_mean) / torch.sqrt(bn_var + eps)
    x_norm = bn_gamma * x_norm + bn_beta
    res = F.conv2d(torch.relu(x_norm), conv_weight, bias = None, stride= 1, padding = 1)
    return res
