import torch

from .base import Config


def defaults() -> Config:
    config = Config(
        # device='cuda' if torch.cuda.is_available() else 'cpu',
        
        # seed=2023, 
        # experiment_name="NAM",
        
        regression=True,
        use_dnn = False, # baseline 
        
        num_epochs=10,
        batch_size=128,
        shuffle=True, # shuffle the training set or not 
        early_stopping_patience=50,  
        decay_rate=0.005, # 0.005
        
        # units_multiplier=2, # adjusted size of the first hidden layer: units_multiplier * unique_value in features 
        
        ## logs
        logdir="output",
        wandb=False, 
        log_loss_frequency=10,
        
        # for tuning
        lr=1e-3,
        l2_regularization=0, # 1e-6
        output_regularization=0, # 1e-3
        dropout=0, # 0.1
        feature_dropout=0,  #0.1
        num_basis_functions=64, # size of the first hidden layer 
        hidden_sizes=[64, 32],  #hidden linear layers' size 
        activation='exu',  ## first hidden layer type; either `exu` or `relu`
    )

    return config