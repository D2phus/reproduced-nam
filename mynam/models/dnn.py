"""DNN baseline"""
import torch 
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(
        self, 
        config, 
        name: str = "DNNModel",
        in_features: int = 1, 
        out_features: int = 1, 
        ) -> None: # type-check
            """
            DNN model as a baseline.
            Args:
            name: identifier for feature net selection
            in_features: size of each input sample
            out_features: size of each output sample

            """
            super(DNN, self).__init__()
            self.config = config
            hidden_sizes = self.config.hidden_sizes
            self.dropout = nn.Dropout(p=self.config.dropout)
            layers = []
            
            layers.append(nn.Linear(in_features, hidden_sizes[0],  bias=True)) # with bias
            layers.append(nn.ReLU())
            layers.append(self.dropout) # dropout 
            
            for in_f, out_f in zip(hidden_sizes[:], hidden_sizes[1:]):
                layers.append(nn.Linear(in_f, out_f,  bias=True))
                layers.append(nn.ReLU())
                layers.append(self.dropout)
                
            layers.append(nn.Linear(hidden_sizes[-1], out_features,  bias=True))
            layers.append(nn.ReLU())
            layers.append(self.dropout)
            
            self.model = nn.Sequential(*layers)
            self.apply(self.initialize_parameters) # note: apply function will recursively applies fn to every submodule
    
    def initialize_parameters(self, m):
        # Xavier initlization 
        if isinstance(m, nn.Linear):
        # if type(m) == nn.Linear: 
            # note that xavier initialization is introduced in the article, while kaiming initlization is adopted here
            # TODO: normal or uniform? why
            torch.nn.init.kaiming_normal_(m.weight)
            # torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, inputs) -> torch.Tensor:
        """
        Args: 
        inputs of shape (batch_size, in_features)
        Returns: 
        outputs of shape (batch_size, out_features)
        """
        return self.model(inputs)