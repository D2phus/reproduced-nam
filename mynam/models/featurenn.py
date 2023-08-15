"""DNN-based sub net for each input feature."""
from .activation import ExU, LinearReLU
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureNN(nn.Module):
    def __init__(
        self, 
        config, 
        name: str, 
        in_features: int,
        num_units: int, 
        feature_index: int, 
        ) -> None: # type-check
            """
            There is a DNN-based sub net for each feature. The first hidden layer is selected amongst:
            1. standard ReLU units
            2. ExU units
            Additionally, dropout layers are added to the end of each hidden layer.
            
            Args:
            in_features: scalar, size of each input sample; default value = 1
            num_units: scalar, number of ExU/LinearReLU hidden units in the first hidden layer 
            feature_index: indicate which feature is learn in this subnet
            """
            super(FeatureNN, self).__init__()
            if config.activation not in ['relu', 'exu', 'gelu', 'elu', 'leakyrelu']:
                raise ValueError('Activation unit should be `gelu`, `relu`, `exu`, `elu`, or `leakyrelu`.')
                
            self.name = name
            self.config = config
            self.in_features = in_features
            self.num_units = num_units
            self.feature_index = feature_index
            # self.dropout = nn.Dropout(p=self.config.dropout)
            self.activation = config.activation
            self.model = self.setup_model()
            
       
    def setup_model(self): 
        layers = list()
        if self.activation == "exu":
            layers.append(ExU(in_features=self.in_features, out_features=self.num_units))
        
        #elif self.activation == 'relu': 
        #    layers.append(LinearReLU(in_features=self.in_features, out_features=self.num_units))
        #layers.append(nn.Dropout(p=self.config.dropout))
        else: 
            if self.activation == 'gelu': 
                activation_cls = nn.GELU
            elif self.activation == 'relu':
                activation_cls = nn.ReLU
            elif self.activation == 'elu': 
                activation_cls = nn.ELU
            elif self.activation == 'leakyrelu':
                activation_cls = nn.LeakyReLU
            layers.append(nn.Linear(self.in_features, self.num_units))
            layers.append(activation_cls())
        
        hidden_sizes = [self.num_units] + self.config.hidden_sizes 
        if len(hidden_sizes) > 1:
            for in_f, out_f in zip(hidden_sizes[:], hidden_sizes[1:]):
                layers.append(LinearReLU(in_features=in_f, out_features=out_f))
                layers.append(nn.Dropout(p=self.config.dropout))
            
        layers.append(nn.Linear(in_features=hidden_sizes[-1], out_features=1)) # output layer; out_features=1
        layers.append(nn.Dropout(p=self.config.dropout))
            
        # gausian layer for uncertainty estimation 
        # layers.append(GaussianLayer(in_features=1))
            
        return nn.Sequential(*layers)
            
            
    def forward(self, inputs) -> torch.Tensor:
        """
        Args: 
        inputs of shape (batch_size): a batch of inputs 
        Return of shape (batch_size, out_features) = (batch_size, 1): a batch of outputs 
        
        """
        outputs = inputs.unsqueeze(1) # TODO: of shape (batch_size, 1)?
        outputs = self.model(outputs)
        return outputs
        # mean, variance = self.model(outputs)
        # return mean, variance