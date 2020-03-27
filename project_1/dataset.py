import numpy as np
from random import gauss
from torch.utils.data import Dataset


class RegressionDataset(Dataset):
    def __init__(self):
        self.x=np.linspace(-3,3,20)
        epsilon=[gauss(0,1) for i in range(20)]
        self.y=2*self.x+epsilon

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,index):
        return self.x[index],self.y[index]
        
