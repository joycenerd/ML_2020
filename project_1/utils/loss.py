import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,yhat,y):
        mse=((yhat - y) ** 2).sum() / self.yhat.data.nelement()
        return torch.sqrt(mse)