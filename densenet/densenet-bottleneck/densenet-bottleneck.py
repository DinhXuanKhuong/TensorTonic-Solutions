import torch
import torch.nn.functional as F

def batch_norm(x, gamma, beta, mean, var, eps):
    x_norm = gamma * (x - mean) / (torch.sqrt(var + eps)) + beta 
    return x_norm

def bottleneck_layer(x, bn1_gamma, bn1_beta, bn1_mean, bn1_var, conv1_weight,
                     bn2_gamma, bn2_beta, bn2_mean, bn2_var, conv2_weight, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, growth_rate, H, W) after the two-stage bottleneck composite.
    """
    # YOUR CODE HERE
    x = torch.as_tensor(x, dtype = torch.float64)
    C = x.shape[1]
    
    bn1_gamma = torch.tensor(bn1_gamma, dtype = torch.float64).reshape(1, C, 1 , 1)
    bn1_beta = torch.tensor(bn1_beta, dtype = torch.float64).reshape(1, C, 1 , 1)
    bn1_mean = torch.tensor(bn1_mean, dtype = torch.float64).reshape(1, C, 1 , 1)
    bn1_var = torch.tensor(bn1_var, dtype = torch.float64).reshape(1, C, 1 , 1)
    
    conv1_weight = torch.tensor(conv1_weight, dtype = torch.float64)
    
    y1 = batch_norm(x, bn1_gamma, bn1_beta, bn1_mean, bn1_var, eps)
    y1 = F.relu(y1)
    y1 = F.conv2d(y1, conv1_weight, padding = 0, bias = None)

    C = y1.shape[1]
    
    bn2_gamma = torch.tensor(bn2_gamma, dtype = torch.float64).reshape(1, C, 1 , 1)
    bn2_beta = torch.tensor(bn2_beta, dtype = torch.float64).reshape(1, C, 1 , 1)
    bn2_mean = torch.tensor(bn2_mean, dtype = torch.float64).reshape(1, C, 1 , 1)
    bn2_var = torch.tensor(bn2_var, dtype = torch.float64).reshape(1, C, 1 , 1)
    
    conv2_weight = torch.tensor(conv2_weight, dtype = torch.float64)
    y2 = batch_norm(y1, bn2_gamma, bn2_beta, bn2_mean, bn2_var, eps)
    y2 = F.relu(y2)
    y2 = F.conv2d(y2, conv2_weight, padding = 1, bias = None)
    return y2
    