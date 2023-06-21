"""Neural additive model"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from typing import Sequence

from .featurenn import FeatureNN

class NAM(nn.Module):
    def __init__(
        self, 
        config, 
        name: str, 
        in_features: int,
        num_units, 
        ) -> None: # type-check
            """
            The neural additive model learns a linear combination of nerual networks each of which attends to a single input feature. The outputs of subnets are added up, with a scalar bias, and passed through a link function for prediction. 
            Args:
            in_features: size of each input sample 
            num_units: number of ExU hidden units for feature subnets. int type when all the feature subnets have the same number of units; list type when the settings are different.
            """
            super(NAM, self).__init__()
            self.config = config
            # a feature NN for each feature
            self.in_features = in_features
            self.num_units = num_units
            self.dropout = nn.Dropout(p=self.config.dropout)
            self.feature_dropout = nn.Dropout(p=self.config.feature_dropout)
            
            # num units for each feature neural net
            if isinstance(self.num_units, list):
                assert len(num_units) == in_features, f"Wrong length of num_units: {len(num_units)}"
            elif isinstance(self.num_units, int):
                self.num_units = [num_units for _ in range(self.in_features)]
                
            # each feature subnet attends to a single input feature
            self.feature_nns = nn.ModuleList([
                FeatureNN(self.config, 
                          name=f"FeatureNN_{feature_index}", 
                          in_features=1, 
                          num_units=self.num_units[feature_index], 
                          feature_index=feature_index) # note the in_features shape 
                for feature_index in range(self.in_features)
            ])
            self.bias = nn.Parameter(data=torch.zeros(1)) # bias of shape (1), initialized with zero
            
    def features_output(self, inputs: torch.Tensor) -> Sequence[torch.Tensor]:
        """
        Return list [torch.Tensor of shape (batch_size, 1)]: the outputs of feature neural nets
        """
        return [self.feature_nns[feature_index](inputs[:, feature_index]) for feature_index in range(self.in_features)] # feature of shape (1, batch_size)
            
    def forward(self, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
        inputs of shape (batch_size, in_features): input samples, 
        
        Returns: 
        nam output of shape (batch_size): add up the outputs of feature nets and bias
        fnn outputs of shape (batch_size, in_features): output of each feature net
        """
        nn_outputs = self.features_output(inputs) # list [Tensor(batch_size,  1)]
        cat_outputs = torch.cat(nn_outputs, dim=-1) # of shape (batch_size, in_features)
        
        dropout_outputs = self.feature_dropout(cat_outputs) # feature dropout
        outputs = dropout_outputs.sum(dim=-1) # sum along the features => of shape (batch_size)
        return outputs + self.bias, dropout_outputs # note that outputs + bias => broadcast