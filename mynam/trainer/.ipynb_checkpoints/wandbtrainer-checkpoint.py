"""trainer class for model training and evaluation"""
from types import SimpleNamespace
from typing import Mapping
from typing import Sequence
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir))) # add parent folder to system paths

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt 

import wandb

from .losses import penalized_loss
from .metrics import accuracy
from .metrics import mae
from .epoch import *
from mynam.models.nam import NAM
from mynam.utils.plotting import *

def wandb_train(config, 
                dataloader_train, 
                dataloader_val, 
                testset: torch.utils.data.Dataset, 
         ):
        """
       tune hyper-parameters with wandb
       https://docs.wandb.ai/guides/sweeps
    
        """
        # initialize sweeps
        run = wandb.init()
        
        # update configurations based on sweeps
        config.update(**wandb.config)
        print(f"Configuration: {config}")
        
        # set up model, optimizer, and criterion
        model = NAM(
          config=config,
          name=f'NAM-{config.activation}',
          in_features=len(testset[0][0]),
          num_units=config.num_basis_functions)
        
        wandb.watch(model, log_freq=config.log_loss_frequency) # watch_gradient 
        print(f"Model summary: {model}")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.decay_rate) 
        criterion = lambda nam_out, fnn_out, model, targets: penalized_loss(config, nam_out, fnn_out, model, targets)
        
        metrics = lambda nam_out, targets: mae(nam_out, targets) if config.regression else accuracy(nam_out, targets)
        metrics_name = "MAE" if config.regression else "Accuracy"
        val_metrics_name = "Val_" + metrics_name
        train_metrics_name = "Train_" + metrics_name
        
        
        # loop the dataset multiple epochs
        for epoch in range(config.num_epochs):
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
                X, y, fnn = testset.tensors 
                f_mu, f_mu_fnn = model(X)
                fig_addi, fig_indiv = plot_mean(X, y, fnn, f_mu, f_mu_fnn, True, True)
                wandb.log({
                    'Additive_fitting': wandb.Image(fig_addi), 
                    'Individual_fitting': wandb.Image(fig_indiv),
                })
                
        # save model in wandb.run.dir, then the file will be uploaded at the end of training.
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.h5'))
        print("Finished Training.")

