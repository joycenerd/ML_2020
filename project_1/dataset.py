import logging
import csv
import numpy as np
from random import gauss
from torch.utils.data import Dataset,DataLoader,Subset
from pathlib import Path
from opt import parse_args

opt=parse_args()

def data_generator(mode,fname,nums):
    if mode=='linear':
        x=np.linspace(-3,3,nums)
        epsilon=[gauss(0,1) for i in range(nums)]
        y=2*x+epsilon

    elif mode=='sin':
        x=np.linspace(0,1,nums)
        epsilon=[gauss(0,0.04) for i in range(nums)]
        y=np.sin(2*np.pi*x)+epsilon

    f=open(fname,'w')
    writer=csv.writer(f)
    for i in range(nums):
        writer.writerow([x[i],y[i]])

class RegressionDataset(Dataset):
    def __init__(self,fname):
        data=np.genfromtxt(fname,delimiter=',')
        self.x=data[:,0]
        self.y=data[:,1]
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,index):
        return self.x[index],self.y[index]