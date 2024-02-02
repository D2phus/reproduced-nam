from typing import Dict

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer, fetch_california_housing, load_diabetes

from .base import LANAMDataset, LANAMSyntheticDataset
from .generator import  *
from mynam.config.default import defaults


cfg = defaults()

def load_breast_data(config=cfg):
    """likelihood: classification."""
    breast_cancer = load_breast_cancer()
    dataset = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
    dataset['target'] = breast_cancer.target
    
    config.likelihood = 'classification'
        
    return LANAMDataset(config,
                        data_path=dataset,
                        features_columns=dataset.columns[:-1],
                        targets_column=dataset.columns[-1])


def load_sklearn_housing_data(config=cfg): 
    """likelihood: regression, N: 20640, D: 8"""
    housing = fetch_california_housing()

    dataset = pd.DataFrame(data=housing.data, columns=housing.feature_names)
    dataset['target'] = housing.target
    
    config.likelihood = 'regression'
     
    return LANAMDataset(config,
                        data_path=dataset,
                        features_columns=dataset.columns[:-1],
                        targets_column=dataset.columns[-1])


def load_diabetes_data(config=cfg): 
    diabetes = load_diabetes()
    dataset = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
    dataset['target'] = diabetes.target
    
    config.likelihood = 'regression'
     
    return LANAMDataset(config,
                        data_path=dataset,
                        features_columns=dataset.columns[:-1],
                        targets_column=dataset.columns[-1])

def load_autompg_data(config=cfg, 
                      data_path: str='LANAM/data/datasets/autompg-dataset/auto-mpg.csv', 
                      features_columns: list = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin'], 
                      targets_column: str = 'mpg'):
    config.likelihood = 'regression'
    
    data = pd.read_csv(data_path)
    data.drop(['car name'], axis=1, inplace=True)
    data['horsepower'].replace('?',np.nan,inplace=True)
    data['horsepower'] = pd.to_numeric(data['horsepower'])
    data.fillna(data['horsepower'].mean(skipna=True),axis=0,inplace=True)
    
    data[targets_column] = data[targets_column]
    return LANAMDataset(config,
                        data_path=data,
                        features_columns=features_columns,
                        targets_column=targets_column)


def load_synthetic_data(config=cfg, 
                   num_samples=1000,
                   generate_functions=synthetic_example(),
                   sigma=1, 
                   sampling_type='uniform'
                   ): 
    """likelihood: regression. 
    Args: 
    -----
    sampling_type: str
        the distribution for X sampling. uniform: U[0, 1]^in_features, normal: N(0.5, 0.2^2) truncated within [0, 1]
    """
    if sampling_type not in ['normal', 'uniform']:
        raise ValueError('Invalid input type for `sampling_type`.')
    config.likelihood = 'regression'
    in_features = len(generate_functions)
    feature_names = [f'X{idx+1}' for idx in range(in_features)]
    
    if sampling_type == 'uniform':
        X = torch.FloatTensor(num_samples, in_features).uniform_(0, 1)
    else: 
        X = torch.zeros(num_samples, in_features)
        torch.nn.init.trunc_normal_(X, mean=0.5, std=0.2, a=0, b=1)
    feature_targets = torch.stack([generate_functions[index](x_i) for index, x_i in enumerate(torch.unbind(X, dim=1))], dim=1) # (batch_size, in_features) 
    y = feature_targets.sum(dim=1) # of shape (batch_size, 1)
    noise = torch.randn_like(y)*sigma
    y += noise
    
    data = pd.DataFrame(X, columns = feature_names)
    data['target'] = pd.Series(y)
    
    return LANAMSyntheticDataset(config,
                        data_path=data,
                        features_columns=data.columns[:-1],
                        targets_column=data.columns[-1], 
                        feature_targets=feature_targets, 
                        sigma=sigma)

    

def load_concurvity_data(config=cfg,
                         num_samples=1000,
                         sigma_1=0.05, 
                         sigma_2=0.5):
    
    """likelihood: regression.
    Simulated dataset built from a known concurvity structure. 
    Refer to Section 6 of the paper: Feature selection algorithms in generalized additive models under concurvity."""
    
    def f4(X, sigma=sigma_1): 
        return torch.pow(X[1], 3) + torch.pow(X[2], 2) + torch.randn(X[1].shape)*sigma
        
    def f5(X, sigma=sigma_1):
        return torch.pow(X[2], 2) + torch.randn(X[2].shape)*sigma
        
    def f6(X, sigma=sigma_1):
        return torch.pow(X[1], 2) + torch.pow(X[3], 3) + torch.randn(X[1].shape)*sigma
        
    def f7(X, sigma=sigma_1):
        return X[0]*X[1] + torch.randn(X[0].shape)*sigma
    
    
    config.likelihood = 'regression'
    in_features = 7
    feature_names = [f'X{idx+1}' for idx in range(in_features)]
    
    X = [torch.FloatTensor(num_samples, 1).uniform_(0, 1) for _ in range(3)]
    generate_functions = [f4, f5, f6, f7]
    for f in generate_functions:
        X.append(f(X, sigma_1))
    X = torch.cat(X, dim=1) 
    feature_targets = torch.zeros_like(X) 
    print(feature_targets.shape)
    feature_targets[:, 0], feature_targets[:, 4], feature_targets[:, 5] = 2*torch.pow(X[:, 0], 2), torch.pow(X[:, 4], 3), 2*torch.sin(X[:, 5])
    y = feature_targets.sum(dim=1)
    y+= torch.randn_like(y)*sigma_2
    
    data = pd.DataFrame(X, columns = feature_names)
    data['target'] = pd.Series(y)
    
    return LANAMSyntheticDataset(config,
                        data_path=data,
                        features_columns=data.columns[:-1],
                        targets_column=data.columns[-1], 
                        feature_targets=feature_targets, 
                        sigma=sigma_2)
