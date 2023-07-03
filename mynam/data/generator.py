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
    return torch.normal(mean=mean, std=0.1)

def generator_1(x):
    #return 1/3*torch.log(100*(x+1)) + 101
    return 1/3*torch.log(100*(x+1)) + 101

def generator_2(x):
    return -8/3*torch.exp(-4*torch.abs(x))

def generator_3(x):
    return torch.sin(10*x)

def generator_4(x):
    return torch.cos(15*x)

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
    task0 = f(x0) + g(x1)
    
    Returns:
    gen_funcs: list of length in_features
    """
    gen_funcs = [generator_1, generator_2]
    gen_funcs_name =  ["generator_1", "generator_2"]
    
    return gen_funcs, gen_funcs_name