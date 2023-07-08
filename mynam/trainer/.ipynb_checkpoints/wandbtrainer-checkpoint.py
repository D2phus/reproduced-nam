"""trainer class for model training and evaluation"""
from types import SimpleNamespace
from typing import Mapping
from typing import Sequence

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt 

from .losses import penalized_loss
from .metrics import accuracy
from .metrics import mae
from .epoch import *

from config import Config

from models.nam import NAM

import os
import sys
sys.path.append(os.path.dirname(os.path.join(os.getcwd()))) 
from utils.plotting import *

import wandb

def sweep_train(config: dict, 
                dataset: torch.utils.data.Dataset, 
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
        model = NAM(config=config, name="NAM_WANDB", in_features=len(dataset[0][0]), num_units=config.num_basis_functions)
        wandb.watch(model, log_freq=config.log_loss_frequency) # watch_gradient 
        print(f"Model summary: {model}")
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.decay_rate) 
        criterion = lambda nam_out, fnn_out, model, targets: penalized_loss(config, nam_out, fnn_out, model, targets)
        
        metrics = lambda nam_out, targets: mae(nam_out, targets) if config.regression else accuracy(nam_out, targets)
        metrics_name = "MAE" if config.regression else "Accuracy"
        val_metrics_name = "val_" + metrics_name
        train_metrics_name = "train_" + metrics_name
        
        dataloader_train, dataloader_val, dataloader_test = dataset.get_dataloaders()
        
        # loop the dataset multiple epochs
        for epoch in range(config.num_epochs):
            # forward + backward + optimize 
            loss_train, metrics_train = train_epoch(criterion, metrics, optimizer, model, dataloader_train)
            loss_val, metrics_val = evaluate_epoch(criterion, metrics, model, dataloader_val)
            
            wandb.log({
                "epoch": epoch, 
                "train_loss": loss_train, 
                "val_loss": loss_val, 
                train_metrics_name: metrics_train, 
                val_metrics_name: metrics_val, # same for swweep configuration and log 
            })
            
            # https://docs.wandb.ai/ref/python/log
            # https://docs.wandb.ai/guides/track/limits, about the log frequency rule
            # fitting for individual features
            if epoch % config.log_loss_frequency == 0: 
                fig = plot_preds(testset, model, epoch)
                wandb.log({
                    "regression": fig
                })
                
            
            
        print("Finished Training.")
        