"""Utilities for logging to the terminal"""
import os 
import time 
import wandb 
from torch.utils.tensorboard import SummaryWriter

from utils.loggers import base

def _format_key(key: str) -> str: 
    """Internal function for formatting keys in Tensorboard format. """
    return key.title().replace('_', '')

class TensorBoardLogger(): 
    """A `TensorBoard` wrapper."""
    def __init__(self, 
                config, 
                project: str = "nam", 
                label: str = "Logs") -> None: 
        self._time = time.time()
        self.label = label 
        self.config = config
        self._iter = 0 
        
        log_dir = os.path.join(config.logdir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        self._summary_writer = SummaryWriter(log_dir=log_dir)
        
        if config.wandb:
            # init a W&B Run object: pass config dictionary with pairs of hyperparameter names and values
            wandb.init(project=project, config=vars(config), reinit=True, magic=True)
        
    def write(self, values: base.LoggingData): 
        for key, value in values.items():
            self._summary_writer.add_scalar(f'{self.label}/{_format_key(key)}', scalar_value=value, global_step=self._iter)
            if self.config.wandb:
                wandb.log({f'{self.label}/{_format_key(key)}':value})
        self._iter += 1