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


project_name = 'NAM-grid-sparse-synthetic'
job_type='sythetic_dataset'
artifact_or_name='synthetic-8:v0'

def setup_dataset(cfg, project_name):
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
    datasets = setup_dataset(cfg, project_name=project_name)