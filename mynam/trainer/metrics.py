import torch 

def rmse(
    logits: torch.Tensor, 
    targets: torch.Tensor
)->torch.Tensor:
    """
    Root mean-squared error for regression 
    """
    criterion = nn.MSELoss()
    loss = torch.sqrt(criterion(logits.view(-1), targets.view(-1)))
    return loss
    
def mae(
    logits: torch.Tensor, 
    targets: torch.Tensor
)->torch.Tensor:
    """
    Mean absolute error 
    """
    return (((logits.view(-1) - targets.view(-1)).abs()).sum() / targets.numel()).item()

def accuracy(
    logits: torch.Tensor, 
    targets: torch.Tensor
)-> torch.Tensor:
    """
    Accuracy for classification
    """
    return (((targets.view(-1) > 0) == (logits.view(-1) > 0.5)).sum() / targets.numel()).item()
    