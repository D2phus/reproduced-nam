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

def generator_1(x):
    #return 1/3*torch.log(100*(x+1)) + 101
    return 8*torch.pow((x-0.5), 2)

def generator_2(x):
    return 0.1 * torch.exp(-8*x+4)
    #return -8/3*torch.exp(-4*torch.abs(x))

def generator_3(x):
    return 5*torch.exp(-2*torch.pow((2*x-1), 2))
    #return torch.sin(10*x)

def generator_4(x):
    return torch.zeros_like(x)

def bernoulli(x, p_start=0.1, p_end=0.9):
    odd = (torch.rand(len(x))*(p_end-p_start) + p_start).float()
    return torch.bernoulli(odd)

def task_0():
    """
    task0 = f(x0) + g(x1) + h(x2) 
    
    Returns:
    gen_funcs: list of length in_features
    """
    gen_funcs = [generator_1, generator_2, generator_3]
    gen_funcs_name =  ["generator_1", "generator_2", "generator_3"]
    return gen_funcs, gen_funcs_name

def task_1():
    """
    task1 = f(x0) + g(x1) + (h(x2)+i(x2))
    
    Returns:
    gen_funcs: list of length in_features
    """
    def in_func_3(x):
        return generator_3(x) + generator_4(x)
    gen_funcs = [generator_1, generator_2, in_func_3]
    gen_funcs_name =  ["generator_1", "generator_2", "generator_3 + generator_4"]
    return gen_funcs, gen_funcs_name

def task_2(): 
    """
    task1 = f(x0) + g(x1) - (h(x2)+i(x2))
    
    Returns:
    gen_funcs: list of length in_features
    """
    def in_func_3(x):
        return -generator_3(x) - generator_4(x)
    gen_funcs = [generator_1, generator_2, in_func_3]
    gen_funcs_name =  ["generator_1", "generator_2", "-(generator_3 + generator_4)"]
    
    return gen_funcs, gen_funcs_name

def task_4():
    """
    task0 = f(x0) + g(x1) + h(x2) + i(x3)
    
    Returns:
    gen_funcs: list of length in_features
    """
    gen_funcs = [generator_1, generator_2, generator_3, generator_4]
    gen_funcs_name =  ["generator_1", "generator_2", "generator_3", "generator_4"]
    return gen_funcs, gen_funcs_name