"""Dataset class for synthetic data"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 

from torch.utils.data import random_split 
class ToyDataset(torch.utils.data.Dataset):
    def __init__(self,
                 task_name: str, 
                 config, 
                 num_samples: int, 
                 in_features: int, 
                 x_start: float, 
                 x_end: float, 
                 gen_funcs,
                 gen_func_names)-> None:
        """
        dataset generated with additive model consisted of synthetic functions. 
        
        Args:
        task_name: indicates the name for this task
        num_samples: the number of samples
        [x_start, x_end]: the x-value range for sampling. X is sampled uniformly.
        in_features: the size of input samples
        gen_funcs: list of synthetic functions for input features
        gen_func_names: list of synthetic function names
        """
        super(ToyDataset, self).__init__()
        self.task_name = task_name
        self.config = config 
        self.num_samples = num_samples
        self.in_features = in_features
        self.gen_funcs = gen_funcs
        self.gen_func_names = gen_func_names
        self.X = torch.stack([torch.linspace(start=x_start, end=x_end, steps=num_samples)]*in_features, dim=1).float()
        self.feature_outs = torch.stack([gen_funcs[index](x_i) for index, x_i in enumerate(torch.unbind(self.X, dim=1))], dim=1) # (batch_size, in_features) 
        self.y = self.feature_outs.sum(dim=1) # of shape (batch_size)
        
        self.setup_dataloaders() 
       
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return self.X[idx], self.y[idx]
    
    def plot(self):
        """
        plot each feature and the additive model
        """
        fig, axs = plt.subplots(1, self.in_features+1, figsize=(10, 2)) 
        for index in range(self.in_features): 
            axs[index].plot(self.X[:, index], self.feature_outs[:, index], '-')
            axs[index].set_title(self.gen_func_names[index])
        axs[-1].plot(self.X[:, 0], self.y, '-')
        axs[-1].set_title(self.task_name)
        
    def setup_dataloaders(self, val_split: float = 0.1, test_split: float = 0.2) -> Tuple[DataLoader, ...]: 
        """
        split the dataset into training set, validation set, and test set
        """
        test_size = int(test_split * len(self))
        val_size = int(val_split * len(self))
        train_size = len(self) - test_size - val_size
        
        train_subset, val_subset, test_subset = random_split(self, [train_size, val_size, test_size])
        
        self.train_dl = DataLoader(train_subset, batch_size=self.config.batch_size, shuffle=True)
        self.val_dl = DataLoader(val_subset, batch_size=self.config.batch_size, shuffle=False)
        self.test_dl = DataLoader(test_subset, batch_size=self.config.batch_size, shuffle=False)
    
    def get_dataloaders(self) -> Tuple[DataLoader, ...]: 
        return self.train_dl, self.val_dl, self.test_dl
        