"""trainer class for model training and evaluation"""
from types import SimpleNamespace
from typing import Mapping
from typing import Sequence

import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from tqdm.autonotebook import tqdm # progress bar
from utils.loggers import TensorBoardLogger

from models.saver import Checkpointer
from .losses import penalized_loss
from .metrics import accuracy
from .metrics import mae

from ray import tune

class Trainer: 
    def __init__(self, 
                config: SimpleNamespace, 
                 model: Sequence[nn.Module], 
                 dataset: torch.utils.data.Dataset
                ) -> None:
        self.config = config
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.decay_rate) 
        self.dataset = dataset
        
        self.writer = TensorBoardLogger(config) 
        self.checkpointer = Checkpointer(model=model, config=config)
        
        if config.wandb:
            wandb.watch(models=model, log='all', log_freq=10)

        self.criterion = lambda nam_out, fnn_out, targets: penalized_loss(config, nam_out, fnn_out, model, targets)
        self.metrics = lambda nam_out, targets: mae(nam_out, targets) if config.regression else accuracy(nam_out, targets)
        self.metrics_name = "MAE" if config.regression else "Accuracy"
        
        self.dataloader_train, self.dataloader_val, self.dataloader_test = self.dataset.get_dataloaders()
        
    def train_step(
        self, 
        batch: torch.Tensor, 
    ) -> torch.Tensor:
        """
        Perform a single mini-batch gredient-descient optimization step.
        """
        features, targets = batch
        
        print(f"features: {features}, targets: {targets}")
        
        self.optimizer.zero_grad()

        preds, fnn_out = self.model(features)
        
        print(f"nam output: {preds}, feature neural nets' output: {fnn_out}")

        loss = self.criterion(preds, fnn_out, targets)
        metrics = self.metrics(preds, targets)
        
        loss.backward()
        
        self.optimizer.step()
        
        return loss, metrics
    

    def train_epoch(
            self,
        ) -> torch.Tensor:
        """
        Perform an epoch of gradient descent optimization on dataloader
        """
        self.model.train()
        loss = 0.0
        metrics = 0.0
            
        with tqdm(self.dataloader_train, leave=False) as pbar:
            for batch in pbar:
                step_loss, step_metrics = self.train_step(batch)
                loss += step_loss
                metrics += step_metrics
                    
                pbar.set_description(f"TL Step: {step_loss: .3f}|{self.metrics_name}:{step_metrics:.3f}")
                    
        return loss / len(dataloader), metrics / len(dataloader)
           

    def evaluate_step(self, batch: torch.Tensor) -> torch.Tensor:

        features, targets = batch

        # Forward pass from the model.
        predictions, fnn_out = self.model(features)

        # Calculates loss on mini-batch.
        loss = self.criterion(predictions, fnn_out, targets)
        metrics = self.metrics(predictions, targets)

        # self.writer.write({"val_loss_step": loss.detach().cpu().numpy().item()})

        return loss, metrics

    def evaluate_epoch(self, dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
        """Performs an evaluation of the `model` on the `dataloader."""
        self.model.eval()
        loss = 0.0
        metrics = 0.0
        with tqdm(dataloader, leave=False) as pbar:
            for batch in pbar:
                # Accumulates loss in dataset.
                with torch.no_grad():
                    # step_loss = self.evaluate_step(model, batch, pbar)
                    # loss += self.evaluate_step(model, batch, pbar)
                    step_loss, step_metrics = self.evaluate_step(self.model, batch)
                    loss += step_loss
                    metrics += step_metrics

                    pbar.set_description((f"VL Step: {step_loss:.3f} | {self.metrics_name}: {step_metrics:.3f}"))

        return loss / len(dataloader), metrics / len(dataloader)
    

    def train(self):
        """
        train models with mini-batch
        """
        num_epochs = self.config.num_epochs
        
        with tqdm(range(num_epochs)) as pbar_epoch:
            for epoch in pbar_epoch:
                # trains model on whole training dataset
                loss_train, metrics_train = self.train_epoch()
                # writes on TensorBoard
                # self.writer.write({
                  #  "loss_train_epoch": loss_train.detach().cpu().numpy().item(),
                  #  f"{self.metrics_name}_train_epoch": metrics_train,
                #})

                # Evaluates model on whole validation dataset, and writes on `TensorBoard`.
                loss_val, metrics_val = self.evaluate_epoch(self.model, self.dataloader_val)
                #self.writer.write({
                 #   "loss_val_epoch": loss_val.detach().cpu().numpy().item(),
                  #  f"{self.metrics_name}_val_epoch": metrics_val,
                #})

                # Checkpoint model weights.
                if epoch % self.config.save_model_frequency == 0:
                    self.checkpointer.save(epoch)

                tune.report(loss=loss_val,)

                # Updates progress bar description.
                pbar_epoch.set_description(f"""Epoch({epoch}):
            TL: {loss_train.detach().cpu().numpy().item():.3f} |
            VL: {loss_val.detach().cpu().numpy().item():.3f} |
            {self.metrics_name}: {metrics_train:.3f}""")
                
    def test(self):
        """
        test models with mini-batch 
        """
        num_epochs = self.config.num_epochs

        with tqdm(range(num_epochs)) as pbar_epoch:
            for epoch in pbar_epoch:

                # Evaluates model on whole validation dataset, and writes on `TensorBoard`.
                loss_test, metrics_test = self.evaluate_epoch(self.model, self.dataloader_test)
                # tune.report(loss_test=loss_test.detach().cpu().numpy().item())
                #self.writer.write({
                 #   "loss_test_epoch": loss_test.detach().cpu().numpy().item(),
                  #  f"{self.metrics_name}_test_epoch": metrics_test,
                #})

                # Updates progress bar description.
                pbar_epoch.set_description("Test Loss: {:.2f} ".format(loss_test.detach().cpu().numpy().item()))