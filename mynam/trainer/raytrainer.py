"""trainer class for model training and evaluation"""
from types import SimpleNamespace
from typing import Mapping
from typing import Sequence

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt 

from ray import tune
from ray.air import Checkpoint, session

from .losses import penalized_loss
from .metrics import accuracy
from .metrics import mae
from .epoch import *

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir))) # add parent folder to system paths
from mynam.models.nam import NAM

def train(config,  
          dataset: torch.utils.data.Dataset
         ):
        """
       train models with mini-batch
       https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
    
        """
        model = NAM(config=config, name="NAM_FINE_TUNE", in_features=len(dataset[0][0]), num_units=config.num_basis_functions)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.decay_rate) 
        criterion = lambda nam_out, fnn_out, model, targets: penalized_loss(config, nam_out, fnn_out, model, targets)
        metrics = lambda nam_out, targets: mae(nam_out, targets) if config.regression else accuracy(nam_out, targets)
        metrics_name = "MAE" if config.regression else "Accuracy"
        
        
        dataloader_train, dataloader_val, dataloader_test = dataset.get_dataloaders()
        
        losses_train = []
        metricses_train = []
        losses_val = []
        metricses_val = []
        
        # loop the dataset multiple epochs
        #for epoch in range(start_epoch, config.num_epochs):
        for epoch in range(10):
            # train for 10 epochs
            # forward + backward + optimize 
            loss_train, metrics_train = train_epoch(criterion, metrics, optimizer, model, dataloader_train)
            loss_val, metrics_val = evaluate_epoch(criterion, metrics, model, dataloader_val)
            
            # save statistics
            losses_train.append(loss_train.detach().cpu().numpy().item())
            metricses_train.append(metrics_train)
            losses_val.append(loss_val.detach().cpu().numpy().item())
            metricses_val.append(metrics_val)
            
            # print statistics
            if epoch % config.log_loss_frequency == 0: 
                print(
                    "epoch=%d, loss_train: %.3f, metrics_train: %.3f"
                    % (epoch + 1, loss_train.detach().cpu().numpy().item(), metrics_train)
                )
                print(
                    "epoch=%d, loss_val: %.3f, metrics_val: %.3f"
                    % (epoch + 1, loss_val.detach().cpu().numpy().item(), metrics_val)
                )
            
            session.report(
                {"loss": loss_val, 
                 metrics_name: metrics_val}, 
            )
        print("Finished Training.")

