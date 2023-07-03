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

from ray import tune
from ray.air import Checkpoint, session

def train(config: dict, 
          dataset: torch.utils.data.Dataset
         ):
        """
       train models with mini-batch
       https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
    
        """
        config = Config(**config)
        model = NAM(config=config, name="NAM_FINE_TUNE", in_features=len(dataset[0][0]), num_units=config.num_basis_functions)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.decay_rate) 
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 180], gamma = 0.9)
        
        criterion = lambda nam_out, fnn_out, model, targets: penalized_loss(config, nam_out, fnn_out, model, targets)
        metrics = lambda nam_out, targets: mae(nam_out, targets) if config.regression else accuracy(nam_out, targets)
        metrics_name = "MAE" if config.regression else "Accuracy"
        
       # ckpts_dir = os.path.join(config.logdir, "ckpts")
        # checkpoint = session.get_checkpoint()
        # if checkpoint:
          #  checkpoint_state = checkpoint.to_dict()
          #  start_epoch = checkpoint_state["epoch"]
          #  model.load_state_dict(checkpoint_state["model_state_dict"])
          #  optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
        #else:
        #    start_epoch = 0
        
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
            
            # learning rate decay 
            #if epoch == 80 or epoch == 100:
               # for p in optimizer.param_groups:
               #     p['lr'] *= 0.1
            #if epoch % 5== 0: 
                #for p in optimizer.param_groups:
                    #p['lr'] *= 0.6
                    
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
            
                # print(f"loss_train_epoch: {loss_train.detach().cpu().numpy().item()}, {self.metrics_name}_train_epoch: {metrics_train}")
                # print(f"loss_val_epoch: {loss_val.detach().cpu().numpy().item()}, {self.metrics_name}_val_epoch: {metrics_val}")
                
            #checkpoint_data = {
             #   "epoch": epoch, 
              #  "model_state_dict": model.state_dict(), 
               # "optimizer_state_dict": optimizer.state_dict(),
            #}
            #checkpoint = Checkpoint.from_dict(checkpoint_data)
            
            session.report(
                {"loss": loss_val, 
                 metrics_name: metrics_val}, 
            )
        print("Finished Training.")
            