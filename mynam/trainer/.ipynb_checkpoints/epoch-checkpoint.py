"""epoch for model training, evaluation, and test"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import penalized_loss
from .metrics import accuracy
from .metrics import mae

from types import SimpleNamespace
from typing import Mapping
from typing import Sequence

def train_epoch(
    criterion, 
    metrics, 
    optimizer: torch.optim.Adam, 
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
) -> torch.Tensor: 
    """
    Perform an epoch of gradient-descent optimization on dataloader 
    """
    model.train()
    avg_loss = 0.0
    avg_metrics = 0.0
            
    for batch in dataloader:
        features, targets = batch

        optimizer.zero_grad()

        preds, fnn_out = model(features)

        step_loss = criterion(preds, fnn_out, model, targets)
        step_metrics = metrics(preds, targets)

        step_loss.backward()
        optimizer.step()
        
        avg_loss += step_loss
        avg_metrics += step_metrics
                
    return avg_loss / len(dataloader), avg_metrics / len(dataloader)
    
    
def  evaluate_epoch(
    criterion, 
    metrics, 
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
) -> torch.Tensor: 
    """
    Perform an epoch of evaluation on dataloader 
    """
    model.eval()
    avg_loss = 0.0
    avg_metrics = 0.0
    for batch in dataloader:
                # Accumulates loss in dataset.
        with torch.no_grad():
            features, targets = batch
    
            preds, fnn_out = model(features)

            step_loss = criterion(preds, fnn_out, model, targets)
            step_metrics = metrics(preds, targets)
            avg_loss += step_loss
            avg_metrics += step_metrics

    return avg_loss / len(dataloader), avg_metrics / len(dataloader)
    