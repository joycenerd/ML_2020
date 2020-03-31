import logging
import csv
import numpy as np
from random import gauss
from torch.utils.data import Dataset,DataLoader,Subset
from pathlib import Path
from opt import parse_args

opt=parse_args()

def data_generator():
    x=np.linspace(-3,3,20)
    epsilon=[gauss(0,1) for i in range(20)]
    y=2*x+epsilon
    f=open('data.csv','w')
    writer=csv.writer(f)
    for i in range(20):
        writer.writerow([x[i],y[i]])

class RegressionDataset(Dataset):
    def __init__(self):
        data=np.genfromtxt('data.csv',delimiter=',')
        self.x=data[:,0]
        self.y=data[:,1]
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,index):
        return self.x[index],self.y[index]
        
class DataSplit():
    def __init__(self,dataset,train_idx,valid_idx,shuffle=False):
        self.dataset=dataset
        self.train_idx=train_idx
        self.valid_idx=valid_idx

        self.train_subset=Subset(self.dataset,self.train_idx)
        self.valid_subset=Subset(self.dataset,self.valid_idx)

    def get_data_loader(self,train_batch_size,valid_batch_size,num_workers=4):
        logging.debug('Initializing train-valid-dataloaders')
        self.train_loader=self.get_train_loader(batch_size=train_batch_size,num_workers=num_workers)
        self.valid_loader=self.get_valid_loader(batch_size=valid_batch_size,num_workers=num_workers)
        return self.train_loader,self.valid_loader
    
    def get_train_loader(self,batch_size,num_workers=4):
        logging.debug('Initializing train loader')
        self.train_loader=DataLoader(self.train_subset,batch_size=batch_size,shuffle=False,num_workers=num_workers)
        return self.train_loader

    def get_valid_loader(self,batch_size,num_workers=4):
        logging.debug('Initializing train loader')
        self.valid_loader=DataLoader(self.valid_subset,batch_size=batch_size,shuffle=False,num_workers=num_workers)
        return self.valid_loader
