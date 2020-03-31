import torch
import copy
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,random_split,Subset
from sklearn.model_selection import LeaveOneOut,KFold
from pathlib import Path
from dataset import data_generator,RegressionDataset
from opt import parse_args

opt=parse_args()

def RMSE_loss(yhat,y):
    # print(yhat)
    # print(y)
    mse=((yhat - y) ** 2).sum() / yhat.data.nelement()
    return torch.sqrt(mse)

def linear_regression(x,weight):
    outputs=[]
    # print(weight)
    # print(x.size())
    for i in x:
        # print(i)
        out=i.view([1,len(i)]).mm(weight)
        outputs.append(out.numpy()) 
    # print(outputs)
    return outputs

def get_best_weight(x,y):
    Q,R=torch.qr(x)
    pseudo_inv=R.pinverse().mm(Q.t())
    weight=torch.mm(pseudo_inv,y)
    return weight

def get_poly_features(x,degree):
    poly_x=[x**n for n in range(1,degree+1)]
    # print

def kf_training(dataset):
    kf=KFold(n_splits=5,random_state=None,shuffle=False)

    total_training_loss=0.0
    total_valid_loss=0.0
    total_weight=np.empty((2,1))

    for train_idx,valid_idx in kf.split(dataset):
        train_set=Subset(dataset,train_idx)
        valid_set=Subset(dataset,valid_idx)

        train_X=[]
        train_Y=[]

        for i,(inputs,labels) in enumerate(train_set):
            train_X.append(inputs)
            train_Y.append(labels)
        newrow=np.ones(len(train_X))
        train_X=np.vstack([train_X,newrow])
        train_X=torch.FloatTensor(train_X)
        train_Y=torch.FloatTensor(train_Y)
        train_X=train_X.t()
        train_Y=train_Y.view([train_Y.size(0),1])
            
        weight=get_best_weight(train_X,train_Y)
        total_weight+=weight.numpy()

        outputs=linear_regression(train_X,weight)
        outputs=torch.FloatTensor(outputs)

        loss=RMSE_loss(outputs,train_Y)
        total_training_loss+=loss/train_X.size(0)

        valid_X=[]
        valid_Y=[]

        for i,(inputs,labels) in enumerate(valid_set):
            valid_X.append(inputs)
            valid_Y.append(labels)
        newrow=np.ones(len(valid_X))
        valid_X=np.vstack([valid_X,newrow])
        valid_X=torch.FloatTensor(valid_X)
        valid_Y=torch.FloatTensor(valid_Y)
        valid_X=valid_X.t()
        valid_Y=valid_Y.view([valid_Y.size(0),1])

        outputs=linear_regression(valid_X,weight)
        outputs=torch.FloatTensor(outputs)

        loss=RMSE_loss(outputs,valid_Y)
        total_valid_loss+=loss/valid_X.size(0)

    best_weight=total_weight/dataset.__len__()
    avg_training_loss=total_training_loss/dataset.__len__()
    avg_valid_loss=total_valid_loss/dataset.__len__()

    print(f'training_loss: {avg_training_loss:.4f}\tvalid_loss: {avg_valid_loss:.4f}\n')

    return best_weight


def loo_train(dataset):
    loo=LeaveOneOut()

    total_training_loss=0.0
    total_valid_loss=0.0
    total_weight=np.empty((2,1))

    for train_idx,valid_idx in loo.split(dataset):
        train_set=Subset(dataset,train_idx)
        valid_set=Subset(dataset,valid_idx)

        train_X=[]
        train_Y=[]

        for i,(inputs,labels) in enumerate(train_set):
            train_X.append(inputs)
            train_Y.append(labels)
        newrow=np.ones(len(train_X))
        train_X=np.vstack([train_X,newrow])
        train_X=torch.FloatTensor(train_X)
        train_Y=torch.FloatTensor(train_Y)
        train_X=train_X.t()
        train_Y=train_Y.view([train_Y.size(0),1])

        # print(train_X) 
        weight=get_best_weight(train_X,train_Y)
        total_weight+=weight.numpy()

        outputs=linear_regression(train_X,weight)
        outputs=torch.FloatTensor(outputs)
        # print(outputs)
        outputs=outputs.view([outputs.size(0),1])

        loss=RMSE_loss(outputs,train_Y)
        total_training_loss+=loss/train_X.size(0)

        valid_X=[]
        valid_Y=[]

        for i,(inputs,labels) in enumerate(valid_set):
            valid_X.append(inputs)
            valid_Y.append(labels)
        newrow=np.ones(len(valid_X))
        valid_X=np.vstack([valid_X,newrow])
        valid_X=torch.FloatTensor(valid_X)
        valid_Y=torch.FloatTensor(valid_Y)
        valid_X=valid_X.t()
        valid_Y=valid_Y.view([valid_Y.size(0),1])

        outputs=linear_regression(valid_X,weight)
        outputs=torch.FloatTensor(outputs)
        outputs=outputs.view([outputs.size(0),1])

        loss=RMSE_loss(outputs,valid_Y)
        total_valid_loss+=loss

    best_weight=total_weight/dataset.__len__()
    avg_training_loss=total_training_loss/dataset.__len__()
    avg_valid_loss=total_valid_loss/dataset.__len__()

    print(f'training_loss: {avg_training_loss:.4f}\tvalid_loss: {avg_valid_loss:.4f}\n')

    return best_weight
        

def main():
    if Path('data.csv').is_file()==False:
        print("yes")
        data_generator()
    data_set=RegressionDataset()
    total_sz=data_set.__len__()
    test_sz=int(0.25*total_sz)
    train_set,test_set=random_split(data_set,[total_sz-test_sz,test_sz])

    weight_loo_deg1=loo_train(train_set)
    weight_kf_deg1=kf_training(train_set)

    # weight_loo-deg5=loo_train(train_set,5)

if __name__=='__main__':
    main()