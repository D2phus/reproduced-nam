import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt 
import plotly.express as px

import numpy as np

import math

from typing import List

 
def plot_recovered_functions(model, X, feature_out, epistemic_uncertainty=True): 
    """plot recovery of known additive structure of dataset.
    Args: 
    model: nn.Module
        `predict` method will give additive and individual predictions with uncertainty estimation. 
    X: (num_samples, in_features)
        input features.
    feature_out: (num_samples, in_features)
        known functions of each feature.
    epistemic_uncertainty: bool
        plot epistemic uncertainty or predictive posterior uncertainty.
    """
    def sort_by_indices(x: torch.Tensor, indices: List):
        """sort x by given indices of the same shape."""
        d1, d2 = x.size()
        ret = torch.stack([x[indices[:, idx], idx] for idx in range(x.shape[1])] ,dim=1)
        return ret
    
    in_features = X.shape[1]
    _, _, f_mu_fnn, f_var_fnn = model.predict(X)
    f_var_fnn = f_var_fnn.flatten(1)
    
    # for visiualization, we sort input and outputs
    X, indices = torch.sort(X, dim=0) # sort input features along each dimension
    # sort known function values, prediction mean, and variance according to permutation indices of X
    feature_out = sort_by_indices(feature_out, indices)
    f_mu_fnn = sort_by_indices(f_mu_fnn, indices)
    f_var_fnn = sort_by_indices(f_var_fnn, indices)
    if not epistemic_uncertainty: 
        f_var_fnn += model.sigma_noise.square().detach().numpy()
    
    # feature-wise residuals are the generated data points with mean contribution of the other feature networks subtracted
    #samples = feature_out.unsqueeze(0)
    #print(samples.shape)
    #num_samples = samples.shape[0]
    #samples = samples.repeat(1, 1, in_features) 
    #residual = torch.stack([torch.cat([f_mu_fnn[:, :idx], f_mu_fnn[:, idx+1:]], dim=1).sum(dim=1) for idx in range(in_features)]).transpose(1, 0).unsqueeze(dim=0)
    #samples -= residual
    # recenter
    #samples -= samples.mean(dim=0).mean(dim=0)# (num_samples, batch_size, in_features)
    #samples = samples.reshape(-1, in_features) 
    
    # re-center the features before visualization
    f_mu_fnn -= f_mu_fnn.mean(dim=0).reshape(1, -1)
    # type and shape formating
    feature_out -= feature_out.mean(dim=0).reshape(1, -1)
    f_mu_fnn, f_var_fnn = f_mu_fnn.detach().numpy(), f_var_fnn.detach().numpy() 
    std_fnn = np.sqrt(f_var_fnn)
    
    feature_out = feature_out.numpy()
    
    cols = 4
    rows = math.ceil(in_features / cols)
    figsize = (2.5*cols ,2*rows)  
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = axs.ravel() 
    for index in range(in_features): 
        lconf, hconf = f_mu_fnn[:, index]-2*std_fnn[:, index], f_mu_fnn[:, index]+2*std_fnn[:, index]
        #customize_ylim = (np.min(feature_out[:, index])-1.5, np.max(feature_out[:, index])+1.5)
        customize_ylim = (-1, 1)
        axs[index].set_ylim(customize_ylim)
        
        plt.setp(axs[index], ylim=customize_ylim)
        axs[index].plot(X[:, index], feature_out[:, index], '--', label="targeted", color="gray")
        axs[index].plot(X[:, index], f_mu_fnn[:, index], '-', label="prediction", color="royalblue")
        #axs[index].scatter(X[:, index].repeat(num_samples), samples[:, index], c='lightgray', alpha=0.3)
        axs[index].fill_between(X[:, index], lconf, hconf, alpha=0.2)

    fig.tight_layout()
    return fig

