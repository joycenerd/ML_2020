import torch
import copy
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
from sklearn.model_selection import LeaveOneOut
from dataset import RegressionDataset
from model import LinearRegressionModel
from utils import parse_args

args=parse_args()

def train():
    data_set=RegressionDataset()
    total_sz=data_set.__len__()
    test_sz=int(0.25*total_sz)
    train_set,test_set=random_split(data_set,[total_sz-test_sz,test_sz])

    input_dim=1
    output_dim=1
    model=LinearRegressionModel(input_dim,output_dim)
    model=model.cuda(args.cuda_devices)
    model.train()

    best_model_param=copy.deepcopy(model.state_dict())
    best_acc=0.0
    num_epochs=args.epoch
    criterion=nn.MSELoss()
    learning_rate=args.lr
    optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

    if args.mode=='leave-one-out':
        loo=LeaveOneOut()
        # for (inputs,labels) in train_set:
            # print(f'{inputs:.4f} {labels:.4f}') 

        for epoch in range(num_epochs):
            train_X=[]
            train_Y=[]
            for inputs,labels in train_set:
                X.append(inputs)
                Y.append(labels)
            for train_index,valid_index in loo.split(train_set):
                train_X=X[train_index]
                train_Y=Y[train_index]
                train_loader=DataLoader(dataset=train_index,batch_size=1,shuffle=False,num_workers=2)
                valid_loader=DataLoader(dataset=valid_index,batch_size=1,shuffle=False,num_workers=1)
                

    

    
if __name__=="__main__":
    train()