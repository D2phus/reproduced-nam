"""trainer class for model training and evaluation"""
import random 
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir))) # add parent folder to system paths

from types import SimpleNamespace
from typing import Mapping
from typing import Sequence
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt 

import wandb

from .losses import penalized_loss
from .metrics import accuracy
from .metrics import mae, mse
from .epoch import *
from mynam.models.nam import NAM
from mynam.utils.plotting import *

import copy


def setup_seeds(seed):
    """Set seeds for everything."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
def train(config, dataloader_train, dataloader_val, testset, ensemble=True, use_wandb=False): 
# get device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if use_wandb:
            # initialize sweeps
            run = wandb.init()
            # update configurations based on sweeps
            config.update(**wandb.config)
            
        print(f"Configuration: {config}")
        # model ensembling
        num_ensemble = config.num_ensemble if ensemble else 1 
        seeds = [*range(num_ensemble)]
        
        # set up criterion and metrics
        criterion = lambda nam_out, fnn_out, model, targets: penalized_loss(config, nam_out, fnn_out, model, targets)
        metrics = lambda nam_out, targets: mse(nam_out, targets) if config.regression else accuracy(nam_out, targets)
        metrics_name = "MSE" if config.regression else "Accuracy"
        val_metrics_name = "Val_" + metrics_name
        train_metrics_name = "Train_" + metrics_name
        
        
        # set up model, optimizer, and criterion
        models = list()
        optimizers = list()
        for idx in range(num_ensemble): 
            setup_seeds(seeds[idx])
            model = NAM(
              config=config,
              name=f'NAM-{config.activation}-{idx}',
              in_features=len(testset[0][0]),
              num_units=config.num_basis_functions).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.decay_rate) 
        
            models.append(model)
            optimizers.append(optimizer)
            
        # save the gradients information of the first model
        #if use_wandb:
        #    for fnn in models[0].feature_nns: 
        #        wandb.watch(fnn, log_freq=config.log_loss_frequency) # log gradients; note that wandb.watch only supports nn.Module object.(not for ModuleList, Tuple, ...)print(f"Model summary: {model}")
        
        # loop the dataset multiple epochs
        for epoch in range(1, config.num_epochs+1):
            # forward + backward + optimize 
            loss_train, metrics_train = ensemble_train_epoch(criterion, metrics, optimizers, models, device, dataloader_train)
            loss_val, metrics_val = ensemble_evaluate_epoch(criterion, metrics, models, device, dataloader_val)
            
            if use_wandb:
                wandb.log({
                    "Train_Loss": loss_train, 
                    "Val_Loss": loss_val, 
                    train_metrics_name: metrics_train, 
                    val_metrics_name: metrics_val, # same for swweep configuration and log 
                })
            else:
                print(f'[EPOCH={epoch}]: Train_Loss: {loss_train}, Val_Loss: {loss_val}, train_metrics_name: {metrics_train}, val_metrics_name: {metrics_val}')
            
            # https://docs.wandb.ai/ref/python/log
            # https://docs.wandb.ai/guides/track/limits, about the log frequency rule
            # fitting for individual features
            if epoch % config.log_loss_frequency == 0: 
                X, y, fnn = testset.tensors 
                pred_map, fnn_map = test(models, X, y, fnn)
                f_mu, f_mu_fnn, f_var, f_var_fnn = pred_map.mean(dim=0), fnn_map.mean(dim=0), pred_map.var(dim=0), fnn_map.var(dim=0) 
                std = np.sqrt(f_var.flatten().detach().numpy())
                
                fig_addi, fig_indiv = plot_uncertainty(X, y, fnn, f_mu, f_var, f_mu_fnn, f_var_fnn, predictive_samples=None, plot_additive=True, plot_individual=True)
                
                if use_wandb:
                    wandb.log({
                        'Output_std_mean': std.mean().item(),
                        'Additive_fitting': wandb.Image(fig_addi), 
                        'Individual_fitting': wandb.Image(fig_indiv),
                    })
                
        # save model in wandb.run.dir, then the file will be uploaded at the end of training.
        if use_wandb:
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
        
        print("Finished Training.")

        

def wandb_train(config, dataloader_train, dataloader_val, dataloader_test, test_samples=None, ensemble=True): 
# get device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # initialize sweeps
        run = wandb.init()
        
        # update configurations based on sweeps
        config.update(**wandb.config)
        print(f"Configuration: {config}")
        # set up criterion and metrics
        criterion = lambda nam_out, fnn_out, model, targets: penalized_loss(config, nam_out, fnn_out, model, targets)
        metrics = lambda nam_out, targets: mse(nam_out, targets) if config.regression else accuracy(nam_out, targets)
        metrics_name = "MSE" if config.regression else "Accuracy"
        val_metrics_name = "Val_" + metrics_name
        train_metrics_name = "Train_" + metrics_name
        
        # model ensembling
        num_ensemble = config.num_ensemble if ensemble else 1 
        seeds = [*range(num_ensemble)]
        
        # set up model, optimizer, and criterion
        models = list()
        optimizers = list()
        for idx in range(num_ensemble): 
            setup_seeds(seeds[idx])
            model = NAM(
              config=config,
              name=f'NAM-{config.activation}-{idx}',
              in_features=len(dataloader_train.dataset[0][0]),
              num_units=config.num_basis_functions).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.decay_rate) 
        
            models.append(model)
            optimizers.append(optimizer)
            
        # save the gradients information of the first model
        for fnn in models[0].feature_nns: 
            wandb.watch(fnn, log_freq=config.log_loss_frequency) # log gradients; note that wandb.watch only supports nn.Module object.(not for ModuleList, Tuple, ...)print(f"Model summary: {model}")
        
        # loop the dataset multiple epochs
        for epoch in range(1, config.num_epochs+1):
            # forward + backward + optimize 
            loss_train, metrics_train = train_epoch(criterion, metrics, optimizer, model, dataloader_train)
            loss_val, metrics_val = evaluate_epoch(criterion, metrics, model, dataloader_val)
            
            wandb.log({
                "Train_Loss": loss_train, 
                "Val_Loss": loss_val, 
                train_metrics_name: metrics_train, 
                val_metrics_name: metrics_val, # same for swweep configuration and log 
            })
            
            # https://docs.wandb.ai/ref/python/log
            # https://docs.wandb.ai/guides/track/limits, about the log frequency rule
            # fitting for individual features
            if epoch % config.log_loss_frequency == 0: 
                features, targets, feature_targets = test_samples
                fig = plot_recovered_functions(model, features, feature_targets)
                wandb.log({
                    'Recovery_functions': wandb.Image(fig),
                })
                
        # save model in wandb.run.dir, then the file will be uploaded at the end of training.
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
        print("Finished Training.")
