"""Generator of synthetic data"""
import torch

from typing import Sequence 
from typing import Callable

def gaussian_noise(x):
    """Generate gaussian noise sampled from N(0, 5/6)"""
    mean = torch.zeros_like(x)
    return torch.normal(mean=mean, std=5/6)

def generator_1(x):
    """Generator function: f(x) = 1/3*log(100x)+101+noise"""
    return 4/3*torch.log(100*(x+2))-1

def generator_2(x):
    """Generator function: g(x) = -4/3*exp(-4*|x|)+noise"""
    return -8/3*torch.exp(-4*torch.abs(x))

def generator_3(x):
    """Generator function: h(x) = sin(10*x)+noise"""
    return torch.sin(10*x)

def generator_4(x):
    """generator function: i(x) = cos(15*x)+noise"""
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
    gen_funcs_name =  ["generator_1", "generator_2", "generator_3+generator_4"]
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
    gen_funcs_name =  ["generator_1", "generator_2", "-(generator_3+generator_4)"]
    
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