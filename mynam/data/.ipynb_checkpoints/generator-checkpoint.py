"""Generator of synthetic data"""
import torch
import math

from typing import Sequence 
from typing import Callable

def gaussian_noise(y):
    """
    Generate noise for the target from the standard normal distribution 
    
    Args:
    y of shape (batch_size): output
    """
    mean = torch.zeros_like(y)
    return torch.normal(mean=mean, std=1)

def pow_shape(x):
    return 8*torch.pow((x-0.5), 2)

def exp_shape1(x):
    return 0.1 * torch.exp(-8*x+4)

def exp_shape2(x):
    return 5*torch.exp(-2*torch.pow((2*x-1), 2))

def zero_shape(x):
    return torch.zeros_like(x)

def bernoulli(x, p_start=0.1, p_end=0.9):
    odd = (torch.rand(len(x))*(p_end-p_start) + p_start).float()
    return torch.bernoulli(odd)

def task_sparse_features():
    """
    Returns:
    gen_funcs: list of length in_features
    """
    gen_funcs = [pow_shape, exp_shape1, exp_shape2, zero_shape, zero_shape, zero_shape, zero_shape, zero_shape]
    gen_funcs_name = [lambda func=x: func.__name__ for x in gen_funcs]
    return gen_funcs, gen_funcs_name