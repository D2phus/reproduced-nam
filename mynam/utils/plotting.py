import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt 
import numpy as np

from typing import Sequence

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
        
        fig, axs = plt.subplots(1, in_features+1, figsize=(10, 2), constrained_layout=True) 
        fig.suptitle(f"epoch={num_epoch}")
        for index in range(in_features): 
            axs[index].plot(X[:, index], feature_outs[:, index], '.', label="targeted")

            axs[index].plot(X[:, index], preds_fnn_outs[:, index], '.', label="predicted")
            axs[index].set_title(gen_func_names[index])
            axs[index].legend()
        axs[-1].plot(X[:, 0], y, '.', label="targeted")
        axs[-1].plot(X[:, 0], preds_out, '.', label="predicted")
        axs[-1].set_title(task_name)
        
        
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