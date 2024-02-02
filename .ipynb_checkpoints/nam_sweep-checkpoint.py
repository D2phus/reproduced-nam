from functools import partial
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from mynam.models.nam import NAM 
from mynam.trainer.wandbtrainer import *
from mynam.config.default import defaults
from mynam.data.toydataset import ToyDataset
from mynam.data.gamdataset import GAMDataset
from mynam.data.generator import *
from mynam.utils.wandb import *
from mynam.utils.plotting import *

import matplotlib.pyplot as plt 
import numpy as np

import wandb 

def setup_dataset(cfg, load_data, project_name):
    if load_data:
        # fetch data and construct dataset from W&B
        processed_datasets = preprocess_and_log(project_name=project_name, job_type='sythetic_dataset', artifact_or_name='synthetic-8:v0')
        trainset = processed_datasets['training']
        valset = processed_datasets['validation']
        testset = processed_datasets['test']
    else:
        # construct dataset from scratch
        gen_funcs, gen_func_names = sparse_task()
        in_features = len(gen_funcs)
        sigma = cfg.prior_sigma_noise
        print(sigma)
        trainset = ToyDataset(gen_funcs, gen_func_names, num_samples=1000, sigma=sigma)
        valset = ToyDataset(gen_funcs, gen_func_names, num_samples=200, sigma=sigma)
        testset = ToyDataset(gen_funcs, gen_func_names, num_samples=50, use_test=True)
        datasets = (trainset, valset, testset)
        load_and_log(datasets, project_name, job_type='synthetic-dataset', artifact_name='synthetic-8', description='sparse synthetic samples with 8 input features, split into train/val/test')
    return trainset, valset, testset

if __name__ == "__main__":
    project_name = 'NAM-grid-sparse-synthetic'
    job_type='sythetic_dataset'
    artifact_or_name='synthetic-8:v0'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation', type=str, help="activation function", default='gelu')
    parser.add_argument('--num_basis_functions', type=int, help="size of the single hidden layer", default=64)
    parser.add_argument('--load_data', type=bool, help="load data from W&B", default=True)
    args = parser.parse_args()
    
    cfg = defaults()
    
    # create W&B run
    wandb.login()
    wandb.finish()
    # setup dataset
    datasets = setup_dataset(cfg, load_data=args.load_data, project_name=project_name)
    trainset, valset, testset = datasets
    
    train_loader, train_loader_fnn = trainset.loader, trainset.loader_fnn
    val_loader, val_loader_fnn = valset.loader, valset.loader_fnn
    X_test, y_test = testset.X, testset.y
    
    parameters_list = {
    'lr': {
        'values': [0.001, 0.01]
    }, 
    'output_regularization': {
        'values': [0, 0.001, 0.1]
    }, 
    'dropout':  {
        'values': [0, 0.2]
    }, 
    'feature_dropout': {
        'values': [0, 0.05]
    }, 
    'activation':  {
        'values': [args.activation]
    }, 
    'num_basis_functions': {
        'values': [args.num_basis_functions]
    }
    
}
    sweep_configuration = {
        'method': 'grid', 
        'name': 'sweep',
        'parameters': parameters_list, 
    }
    # initialize the sweep 
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, 
        project=project_name,
    )
    # training
    wandb.agent(sweep_id, 
            function=partial(train, 
                             config=cfg, 
                             dataloader_train=train_loader, 
                             dataloader_val=val_loader, 
                             testset=testset,
                             ensemble=True, 
                             use_wandb=True))