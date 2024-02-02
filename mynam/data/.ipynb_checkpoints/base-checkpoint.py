"""Dataset class for synthetic data"""
import math 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from typing import Tuple, Sequence, Union

from nam.data import transform_data
from nam.data import CSVDataset


class LANAMDataset(CSVDataset): 
    """https://github.com/AmrMKayid/nam/blob/main/nam/data/utils.py"""
    def __init__(self, config, data_path, features_columns, targets_column, weights_column=None):
        super().__init__(config=config,
                         data_path=data_path,
                         features_columns=features_columns,
                         targets_column=targets_column,
                         weights_column=weights_column)
        self.raw_data = data_path
        
        self.col_min_max = self.get_col_min_max()

        self.features, self.features_names = transform_data(self.raw_X)
        self.compute_features()
        self.in_features = len(self.features_names)

        if (config.likelihood == 'classification') and (not isinstance(self.raw_y, np.ndarray)):
            targets = pd.get_dummies(self.raw_y).values
            targets = np.array(np.argmax(targets, axis=-1))
        else:
            targets = self.y

        self.features = torch.from_numpy(self.features).float().to(config.device)
        self.targets = torch.from_numpy(targets).view(-1, 1).float().to(config.device)
        self.wgts = torch.from_numpy(self.wgts).to(config.device)

        self.setup_dataloaders()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[np.array, ...]:
        return self.features[idx], self.targets[idx]  #, self.wgts[idx]

    def get_col_min_max(self):
        col_min_max = {}
        for col in self.raw_X:
            unique_vals = self.raw_X[col].unique()
            col_min_max[col] = (np.min(unique_vals), np.max(unique_vals))

        return col_min_max

    def compute_features(self):
        single_features = np.split(np.array(self.features), self.features.shape[1], axis=1)
        self.unique_features = [np.unique(f, axis=0) for f in single_features]

        self.single_features = {col: sorted(self.raw_X[col].to_numpy()) for col in self.raw_X}
        self.ufo = {col: sorted(self.raw_X[col].unique()) for col in self.raw_X}

    def setup_dataloaders(self, val_split: float = 0.1, test_split: float = 0.2) -> Tuple[DataLoader, ...]:
        def setup_feature_dataloaders(in_features, subset):
            dl_fnn = list()
            for idx in range(in_features):
                features, targets = subset[:]
                feature = features[:, idx].reshape(-1, 1) # (num_samples, 1)
                targets = targets.reshape(-1, 1)
                dataset = TensorDataset(feature, targets)
                loader = DataLoader(dataset, batch_size=self.config.batch_size)
                dl_fnn.append(loader)
                
            return dl_fnn
                
        test_size = int(test_split * len(self))
        val_size = int(val_split * (len(self) - test_size))
        train_size = len(self) - val_size - test_size

        train_subset, val_subset, test_subset = random_split(self, [train_size, val_size, test_size])
    
        self.train_dl_fnn = setup_feature_dataloaders(self.in_features, train_subset)
        
        self.val_dl_fnn = setup_feature_dataloaders(self.in_features, val_subset)
        
        self.test_dl_fnn = setup_feature_dataloaders(self.in_features, test_subset)

        self.train_dl = DataLoader(train_subset, batch_size=self.config.batch_size, shuffle=True)

        self.val_dl = DataLoader(val_subset, batch_size=self.config.batch_size, shuffle=False)

        self.test_dl = DataLoader(test_subset, batch_size=self.config.batch_size, shuffle=False)
        
    def train_dataloaders(self) -> Tuple[DataLoader, ...]:
        return self.train_dl, self.train_dl_fnn, self.val_dl, self.val_dl_fnn

    def test_dataloaders(self) -> DataLoader:
        return self.test_dl, self.test_dl_fnn
    
    def plot_scatterplot_matrix(self):
        """Plot scatterplot matrix on test set."""
        subset = self.test_dl.dataset
        indices = subset.indices
        features = self.features[indices]
        y = self.y[indices]
            
        cols = self.in_features + 1
        rows = cols
        figsize = (1.5*cols ,1.5*rows)
        fig, axs = plt.subplots(rows, cols, figsize=figsize)
        axs = axs.ravel() # 
        #text_kwargs = dict(ha='center', va='center', fontsize=28)
        for index in range(self.in_features): 
            for idx in range(self.in_features):
                #print(index, idx)
                ax = axs[index*cols+idx]
                if index == idx:
                    ax.set_title(f'{self.features_names[idx]}')
                    continue
                ax.plot(features[:, idx], features[:, index], '.', color='royalblue')
            axs[cols*(index+1)-1].plot(y, features[:, index], '.', color='royalblue')
                #axs[index].set_title(f"X{index}")
        for idx in range(self.in_features):
            axs[cols*(rows-1)+idx].plot(features[:, idx], y, '.', color='royalblue')
        axs[cols*rows-1].set_title('y')
        fig.tight_layout()


class LANAMSyntheticDataset(LANAMDataset):
    def __init__(self, config, data_path, features_columns, targets_column, feature_targets, sigma, weights_column=None):
        self.feature_targets = feature_targets
        self.sigma = sigma
        
        super().__init__(config=config,
                         data_path=data_path,
                         features_columns=features_columns,
                         targets_column=targets_column,
                         weights_column=weights_column)
        
    def plot_dataset(self, subset:
torch.utils.data.Subset=None):
        """
        plot each features on dataset.
        """
        cols = 4
        rows = math.ceil(self.in_features / cols)
        figsize = (2*cols ,2*rows)
        fig, axs = plt.subplots(rows, cols, figsize=figsize)
        axs = axs.ravel()  
        if subset is None:
            features = self.features
            feature_targets = self.feature_targets 
        else:
            indices = subset.indices
            features = self.features[indices]
            feature_targets = self.feature_targets[indices]
            
        for index in range(self.in_features): 
            axs[index].plot(features[:, index], feature_targets[:, index], '.', color='royalblue')  
            axs[index].set_xlabel(self.features_names[index])
            axs[index].set_ylabel(f'f{index+1}')
            
        fig.tight_layout()
     
    def get_test_samples(self): 
        """
        get all the samples in testset.
        """
        test_subset = self.test_dl.dataset
        indices = test_subset.indices
        features, targets = test_subset[:]
        feature_targets = self.feature_targets[indices, :]
        return features, targets, feature_targets
        
    def setup_dataloaders(self, val_split: float = 0.1, test_split: float = 0.2) -> Tuple[DataLoader, ...]:
        def setup_feature_dataloaders(in_features, subset, feature_targets):
            dl_fnn = list()
            for idx in range(in_features):
                features, _ = subset[:]
                feature = features[:, idx].reshape(-1, 1) # (num_samples, 1)
                targets = feature_targets[:, idx].reshape(-1, 1)
                dataset = TensorDataset(feature, targets)
                loader = DataLoader(dataset, batch_size=self.config.batch_size)
                dl_fnn.append(loader)
                
            return dl_fnn
        
        test_size = int(test_split * len(self))
        val_size = int(val_split * (len(self) - test_size))
        train_size = len(self) - val_size - test_size

        train_subset, val_subset, test_subset = random_split(self, [train_size, val_size, test_size])
    
        self.train_dl_fnn = setup_feature_dataloaders(self.in_features, train_subset, self.feature_targets[train_subset.indices])
        
        self.val_dl_fnn = setup_feature_dataloaders(self.in_features, val_subset, self.feature_targets[val_subset.indices])
        
        self.test_dl_fnn = setup_feature_dataloaders(self.in_features, test_subset, self.feature_targets[test_subset.indices])

        self.train_dl = DataLoader(train_subset, batch_size=self.config.batch_size, shuffle=True)

        self.val_dl = DataLoader(val_subset, batch_size=self.config.batch_size, shuffle=False)

        self.test_dl = DataLoader(test_subset, batch_size=self.config.batch_size, shuffle=False)


class GAMDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, fnn, batch_size=64, use_test=False)-> None:
        """
        dataset for generalized additive models.
        
        Args:
        -----------
        X of shape (batch_size, in_features)
        y of shape (batch_size)
        fnn of shape (batch_size, in_features): output for each feature dimension
        batch_size, int
        use_test, bool: the dataloader won't be shuffled if `use_test` is True. 
        
        """
        super(GAMDataset, self).__init__()
        self.X = X
        self.y = y
        self.fnn = fnn
        self.batch_size = batch_size
        self.use_test = use_test
        self.in_features = X.shape[1]
        
        self.get_loaders()
       
    
    @property
    def tensors(self):
        """Returns data."""
        return self.X, self.y, self.fnn
    
    
    def __len__(self):
        return len(self.X)

    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return self.X[idx], self.y[idx]

    
    def plot(self, additive=True):
        cols = 4
        rows = math.ceil(self.in_features / cols)
        figsize = (2*cols ,2*rows)
        fig, axs = plt.subplots(rows, cols, figsize=figsize)
        axs = axs.ravel() # 
        fig.tight_layout()
        for index in range(self.in_features): 
            axs[index].plot(self.X[:, index], self.fnn[:, index], '.', color='royalblue')
            axs[index].set_title(f"X{index}")
        
        if additive:
            fig, axs = plt.subplots()
            axs.plot(self.X[:, 0], self.y, '.', color='royalblue')
    
    
    def get_loaders(self): 
        """
        Returns:
        ---------------
        loader 
        loader_fnn, list
        """
        dataset = TensorDataset(self.X, self.y) # X, y: of shape (batch_size, 1)
        dataset_fnn = [TensorDataset(self.X[:, index].reshape(-1, 1), self.fnn[:, index].reshape(-1, 1)) for index in range(self.in_features)]
        
        shuffle = False if self.use_test else True
        self.loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        self.loader_fnn = [DataLoader(dataset_fnn[index], batch_size=self.batch_size, shuffle=shuffle) for index in range(self.in_features)]

     
