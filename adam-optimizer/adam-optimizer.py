import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    # pass
    m = np.array(m)
    v = np.array(v)
    grad = np.array(grad)
    param = np.array(param)
    m_new = beta1 * m + (1 - beta1)*grad 
    v_new = beta2 * v + (1 - beta2)*(grad**2)

    m_new_norm = m_new / (1 - beta1 ** t)
    v_new_norm = v_new / (1 - beta2 ** t)
    
    param_new =  param - lr *(m_new_norm / (v_new_norm**0.5 + eps))

    return param_new, m_new, v_new