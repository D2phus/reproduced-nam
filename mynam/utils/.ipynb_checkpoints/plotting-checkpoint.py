import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt 
import numpy as np

from typing import Sequence

import math

def plot_ensemble(dataset: torch.utils.data.Dataset, 
               models: nn.Module, 
               num_epoch: int):
    """
    plot the mean and variances of individual fits given by the models on the dataset
    """
    num_models = len(models) # number of models 
    
    task_name = dataset.task_name
    in_features = dataset.in_features
    X, y, feature_outs, gen_func_names = dataset.X, dataset.y, dataset.feature_outs, dataset.gen_func_names
    
    outputs = []
    fnn_outputs = []
    for index in range(num_models): 
        out, fnn_out = models[index](X)
        out, fnn_out = out.detach().numpy(), fnn_out.detach().numpy()
        
        outputs.append(out)
        fnn_outputs.append(fnn_out)
        
    outputs = np.stack(outputs) # (num_models, num_samples)
    fnn_outputs = np.stack(fnn_outputs) 
    
    # compute the mean and standard deviance along models
    mean = np.mean(fnn_outputs, axis=0)
    std = np.std(fnn_outputs, axis=0)
    # compute the confidence interval: mean +- 2*std
    upper_bound = mean + 2*std
    bottom_bound = mean - 2*std
    
    cols = 4
    rows = math.ceil(in_features / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(16, 6))
    fig.tight_layout()
    for index in range(in_features): 
        col = index % cols 
        row = math.floor(index / cols)
        axs[row, col].plot(X[:, index], feature_outs[:, index], '--', label="targeted", color="gray")
        axs[row, col].plot(X[:, index], mean[:, index], '-', label="mean predictions", color="royalblue")
        
        axs[row, col].fill_between(X[:, index], upper_bound[:, index], bottom_bound[:, index], alpha=0.2)
        
        axs[row, col].set_title(f"X{index}")
            
        axs[row, col].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    

def plot_preds(dataset: torch.utils.data.Dataset, 
               model: nn.Module, 
               num_epoch: int):
        """
        Plot the fitting of individual fits on the dataset when training epochs = num_epoch
        """
        task_name = dataset.task_name
        in_features = dataset.in_features
        X, y, feature_outs, gen_func_names = dataset.X, dataset.y, dataset.feature_outs, dataset.gen_func_names
        
        preds_out, preds_fnn_outs = model(X)
        preds_out, preds_fnn_outs = preds_out.detach().numpy(), preds_fnn_outs.detach().numpy()
        
        cols = 4
        rows = math.ceil(in_features / cols)
        fig, axs = plt.subplots(rows, cols)
        fig.tight_layout()
        for index in range(in_features): 
            col = index % cols 
            row = math.floor(index / cols)
            axs[row, col].plot(X[:, index], feature_outs[:, index], '--', label="targeted")
            axs[row, col].plot(X[:, index], preds_fnn_outs[:, index], '-', label="predicted")
            axs[row, col].set_title(f"X{index}")
        
        #fig, axs = plt.subplots(1, in_features, figsize=(15, 2))
        ## axs.set_box_aspect(0.8)
        #for index in range(in_features): 
        #    axs[index].plot(X[:, index], feature_outs[:, index], '--', label="targeted")
        #    axs[index].plot(X[:, index], preds_fnn_outs[:, index], '-', label="predicted")
        #    axs[index].set_title(f'X{index}')

        #axs[-1].plot(X[:, 0], y, '.', label="targeted")
        #axs[-1].plot(X[:, 0], preds_out, '.', label="predicted")
        #axs[-1].set_title(task_name)
        
        return fig
        
        
def plot_training(num_epochs: int, 
                  losses_train: Sequence, 
                  metricses_train: Sequence, 
                  losses_val: Sequence, 
                  metricses_val: Sequence):
    """
    Plot the training & validation loss and metrics
    """
    print(f"The minimum validation loss: {min(losses_val)}")
    print(f"The minimum validation metrics: {min(metricses_val)}")
    x = np.arange(num_epochs)
    fig = plt.figure()
    plt.plot(x, losses_train, '-', label="train loss")
    plt.plot(x, metricses_train, '-', label="train metrics")
    plt.plot(x, losses_val, '-', label="validation loss")
    plt.plot(x, metricses_val, '-', label="validation metrics")
    plt.legend()