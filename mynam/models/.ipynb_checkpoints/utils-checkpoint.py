import torch 
import numpy as np

def get_num_units( 
    config, 
    features: torch.Tensor):
    """
    get number of units in the exu/relu layer 
    """
    num_unique_vals = [len(np.unique(features[:, feature_index])) for feature_index in range(features.shape[1])]
    
    num_units = [min(config.num_basis_functions, feature_index * config.units_multiplier) for feature_index in num_unique_vals]
    
    return num_units