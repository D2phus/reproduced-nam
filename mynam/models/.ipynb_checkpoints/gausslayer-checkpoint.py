import torch 
import torch.nn as nn
import torch.nn.functional as F


class GaussianLayer(nn.Module):
    def __init__(
        self, 
        in_features: int = 1, 
        ) -> None: # type-check
            """
            A final layer which outputs two values, corresponding to the predicted mean and variance.
            Args:
            in_features: size of each layer input 
            
            https://arxiv.org/abs/1612.01474
            
            """
            super(GaussianLayer, self).__init__()
            self.in_features = in_features
            # parameters for mean
            self.w1 = nn.Parameter(torch.Tensor(in_features))
            self.b1 = nn.Parameter(torch.Tensor(in_features)) # note; a bias for each feature
            # parameters for variance
            self.w2 = nn.Parameter(torch.Tensor(in_features))
            self.b2 = nn.Parameter(torch.Tensor(in_features)) # note; a bias for each feature
            
            self.initialize_parameters()  
            
    def initialize_parameters(self):
        """
        Initializing the parameters. 
        - weights: regular, xavier uniform(why uniform?)
        - bias: N(0, 0.5²)
        """
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        
        bias_mean = torch.zeros(self.in_features)
        self.b1 = nn.Parameter(torch.normal(mean=bias_mean, std=0.5))
        self.b2 = nn.Parameter(torch.normal(mean=bias_mean, std=0.5))
        
    def forward(self, inputs) -> torch.Tensor:
        """
        Args: 
        inputs of shape (batch_size, in_features)
        Returns: 
        predicted mean of shape (batch_size, 1) 
        predicted variance of shape (batch_size, 1)
        """
        mean = torch.matmul((inputs-self.b1), self.w1)
        variance = torch.matmul((inputs-self.b2), self.w2)
        # enforce the positivity constraint on the variance
        variance = torch.log(1+torch.exp(variance)) + 1e-06 # positive 
        return mean, variance