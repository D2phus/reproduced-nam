import torch

from .base import Config


def defaults() -> Config:
    config = Config(
        experiment_name="NAM-grid-1",
        seed=2023, 
        
        prior_sigma_noise=0.7,
        
        regression=True,
        use_dnn = False, # baseline 
        
        num_epochs=100,
        batch_size=128,
        shuffle=True, # shuffle the training set or not 
        early_stopping_patience=50,  
        decay_rate=0.005, # 0.005
        
        ## logs
        logdir="./output",
        wandb=False, 
        log_loss_frequency=20,
        
        # for tuning
        lr=1e-3,
        l2_regularization=1e-5, 
        output_regularization=0, # 1e-3
        dropout=0, # 0.1
        feature_dropout=0,  #0.1
        num_basis_functions=64, # size of the first hidden layer 
        hidden_sizes=[],  #hidden linear layers' size 
        activation='relu',  ## hidden unit type
        
        num_ensemble=10, 
        
    )

    return config