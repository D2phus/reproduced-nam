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

from config import Config

import os
import sys
sys.path.append(os.path.dirname(os.path.join(os.getcwd()))) 
from utils.plotting import *
from models.nam import NAM


class Trainer: 
    def __init__(self, 
                config: SimpleNamespace, 
                 model: Sequence[nn.Module], 
                 dataset: torch.utils.data.Dataset,
                ) -> None:
        """
        """
        self.config = config
    
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.decay_rate) 
        self.dataset = dataset
    
        self.criterion = lambda nam_out, fnn_out, model, targets: penalized_loss(self.config, nam_out, fnn_out, model, targets)
        self.metrics_name = "MAE" if config.regression else "Accuracy"
        self.metrics = lambda nam_out, targets: mae(nam_out, targets) if config.regression else accuracy(nam_out, targets)
        
        self.dataloader_train, self.dataloader_val, self.dataloader_test = self.dataset.get_dataloaders()
        
    
    def train_step(
        self, 
        model: nn.Module, 
        batch: torch.Tensor, 
    ) -> torch.Tensor:
        """
        Perform a single mini-batch gredient-descient optimization step.
        """
        features, targets = batch
        self.optimizer.zero_grad()

        preds, fnn_out = model(features)
        
        loss = self.criterion(preds, fnn_out, model, targets)
        metrics = self.metrics(preds, targets)
        
        loss.backward()
        
        self.optimizer.step()
        
        return loss, metrics
    

    def evaluate_step(self, 
                      model: nn.Module, 
                      batch: torch.Tensor) -> torch.Tensor:

        features, targets = batch

        # Forward pass from the model.
        predictions, fnn_out = model(features)

        # Calculates loss on mini-batch.
        loss = self.criterion(predictions, fnn_out, model, targets)
        metrics = self.metrics(predictions, targets)

        return loss, metrics

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
            loss_train, metrics_train = train_epoch(self.criterion, self.metrics, self.optimizer, self.model, self.dataloader_train)
            loss_val, metrics_val = evaluate_epoch(self.criterion, self.metrics, self.model, self.dataloader_val)
            
            
            # save statistics
            losses_train.append(loss_train.detach().cpu().numpy().item())
            metricses_train.append(metrics_train)
            losses_val.append(loss_val.detach().cpu().numpy().item())
            metricses_val.append(metrics_val)
            
            # print statistics
            if epoch % self.config.log_loss_frequency == 0:
                print(f"loss_train_epoch: {loss_train.detach().cpu().numpy().item()}, {self.metrics_name}_train_epoch: {metrics_train}")
                print(f"loss_val_epoch: {loss_val.detach().cpu().numpy().item()}, {self.metrics_name}_val_epoch: {metrics_val}")
                if epoch % 20 == 0:
                    plot_preds(self.dataset, self.model, epoch)
                
        plot_preds(self.dataset, self.model, num_epochs)
        plot_training(num_epochs, losses_train, metricses_train, losses_val, metricses_val)
        return losses_train, metricses_train, losses_val, metricses_val
    
    
    def test(self, testset):
        """
        test models with mini-batch 
        """
        num_epochs = self.config.num_epochs
        test_dl = DataLoader(testset, batch_size=self.config.batch_size, shuffle=False)
        #with tqdm(range(num_epochs)) as pbar_epoch:
         #   for epoch in pbar_epoch:
        for epoch in range(num_epochs):
            loss_test, metrics_test = evaluate_epoch(self.criterion, self.metrics, self.model, test_dl)
        print(f"loss_test_epoch: {loss_test.detach().cpu().numpy().item()}, {self.metrics_name}_test_epoch: {metrics_test}")
        plot_preds(testset, self.model, num_epochs)
        

        