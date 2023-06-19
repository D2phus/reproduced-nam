"""Utilities for logging to Weights & Biases. """

import torch 
import wandb

from utils.loggers import base

class WandBLogger(base.Logger): 
    """Log to a `wandb` dashboard."""
    
    def __init__(self, 
                project: str="nam",
                config: dict=None)->None: 
        super().__init__(log_dir=config.logdir)
        wandb.init(project=project, config=config)
        
    def write(self, data:base.LoggingData) -> None: 
        wandb.log(data)
    
    def watch(self, 
             model: torch.nn.Module, 
             **kwargs: dict)->None: 
        wandb.watch(model, kwargs)