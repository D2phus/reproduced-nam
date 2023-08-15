"""trainer class for model training and evaluation"""
from types import SimpleNamespace
from typing import Mapping
from typing import Sequence

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt 

from .losses import penalized_loss
from .metrics import accuracy
from .metrics import mae
from .epoch import *


class Trainer: 
    def __init__(self, 
                config: SimpleNamespace, 
                 model: Sequence[nn.Module], 
                 trainset: torch.utils.data.dataset,
                 valset: torch.utils.data.dataset, 
                 testset=None
                ) -> None:
        self.config = config
    
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.decay_rate) 
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=0.995, step_size=1)
        self.trainset = trainset
        self.valset = valset
        self.testset = testset
        
        self.dataloader_train = trainset.loader
        self.dataloader_val = valset.loader
        if testset is not None: 
            self.dataloader_test = testset.loader
            
    
        self.criterion = lambda nam_out, fnn_out, model, targets: penalized_loss(self.config, nam_out, fnn_out, model, targets)
        self.metrics_name = "MAE" if config.regression else "Accuracy"
        self.metrics = lambda nam_out, targets: mae(nam_out, targets) if config.regression else accuracy(nam_out, targets)

    def train(self):
        """
       train models with mini-batch
    
        """
        num_epochs = self.config.num_epochs
        
        losses_train = []
        metricses_train = []
        losses_val = []
        metricses_val = []
        
        for epoch in range(num_epochs):
        
            # trains model on whole training dataset, compute the training and validation loss & metric
            loss_train, metrics_train = train_epoch(self.criterion, self.metrics, self.optimizer, self.model, self.dataloader_train, self.scheduler)
            loss_val, metrics_val = evaluate_epoch(self.criterion, self.metrics, self.model, self.dataloader_val)
            
            
            # save statistics
            losses_train.append(loss_train.detach().cpu().numpy().item())
            metricses_train.append(metrics_train)
            losses_val.append(loss_val.detach().cpu().numpy().item())
            metricses_val.append(metrics_val)
            
            # print statistics
            if epoch % self.config.log_loss_frequency == 0:
                print(f"==============EPOCH {epoch}================")
                print(f"loss_train_epoch: {loss_train.detach().cpu().numpy().item()}, {self.metrics_name}_train_epoch: {metrics_train}")
                print(f"loss_val_epoch: {loss_val.detach().cpu().numpy().item()}, {self.metrics_name}_val_epoch: {metrics_val}")
                
        return losses_train, metricses_train, losses_val, metricses_val
    
    
    def test(self):
        """
        test models with mini-batch 
        """
        if self.testset is not None: 
            num_epochs = self.config.num_epochs
            #with tqdm(range(num_epochs)) as pbar_epoch:
             #   for epoch in pbar_epoch:
            for epoch in range(num_epochs):
                loss_test, metrics_test = evaluate_epoch(self.criterion, self.metrics, self.model, self.dataloader_test)
            print(f"loss_test_epoch: {loss_test.detach().cpu().numpy().item()}, {self.metrics_name}_test_epoch: {metrics_test}")



