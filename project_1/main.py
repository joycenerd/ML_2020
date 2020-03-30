import torch
import copy
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,random_split
from sklearn.model_selection import LeaveOneOut,KFold
from torch.autograd import Variable
from pathlib import Path
from dataset import RegressionDataset,DataSplit
from model import LinearRegressionModel
from opt import parse_args
from utils.loss import RMSELoss


opt=parse_args()

##########          Testing          ##########
def test(test_set,model):
    test_loader=DataLoader(test_set,batch_size=1,num_workers=1)

    criterion=RMSELoss()
    testing_loss=0.0

    model=model.cuda(opt.cuda_devices)
    model.eval()

    for i,(inputs,labels) in enumerate(test_loader):
        inputs=Variable(inputs.cuda(opt.cuda_devices))
        labels=Variable(labels.cuda(opt.cuda_devices))
        inputs=inputs.view([inputs.size(0),1])
        labels=labels.view([labels.size(0),1])


        outputs=model(inputs)
        loss=criterion(outputs,labels)

        testing_loss+=loss.item()*inputs.size(0)

    testing_loss=testing_loss/(len(test_loader)*1)
    print(f'testing_loss: {testing_loss:.4f}')

def loss_visualization(training_loss,valid_loss,fname):
    plt.figure(figsize=[8,6])
    plt.plot(training_loss,'r',linewidth=3.0)
    plt.plot(valid_loss,'b',linewidth=3.0)
    plt.legend(['Training loss','Validation loss'],fontsize=18)
    plt.xlabel('Epochs',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    plt.savefig(Path("./figure").joinpath(fname))

def five_fold_training(train_set,model):
    criterion=RMSELoss()
    learning_rate=opt.lr
    optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

    kf=KFold(n_splits=5,random_state=None,shuffle=False)
    fold_sz=int(train_set.__len__()*0.2)

    total_training_loss=0.0
    total_valid_loss=0.0

    for train_index,valid_index in kf.split(train_set):
        split=DataSplit(train_set,train_index,valid_index)
        train_loader,valid_loader=split.get_data_loader(fold_sz*4,fold_sz)

        training_loss=0.0
        valid_loss=0.0

        for i,(inputs,labels) in enumerate(train_loader):
            inputs=Variable(inputs.cuda(opt.cuda_devices))
            inputs=inputs.view([inputs.size(0),1])
            labels=Variable(labels.cuda(opt.cuda_devices))
            labels=labels.view([labels.size(0),1])

            optimizer.zero_grad()
            outputs=model(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            training_loss+=loss.item()*inputs.size(0)
        
        training_loss=training_loss/(len(train_loader)*inputs.size(0))
        total_training_loss+=training_loss

        model.eval()

        for i,(inputs,labels) in enumerate(valid_loader):
            inputs=inputs.cuda(opt.cuda_devices)
            labels=labels.cuda(opt.cuda_devices)
            inputs=inputs.view([inputs.size(0),1])
            labels=labels.view([labels.size(0),1])

            outputs=model(inputs)
            loss=criterion(outputs,labels)

        valid_loss=loss.item()/len(valid_loader)
        total_valid_loss+=valid_loss
    
    avg_training_loss=total_training_loss/5
    avg_valid_loss=total_valid_loss/5

    return model,avg_training_loss,avg_valid_loss

def loo_training(train_set,model):
    criterion=RMSELoss()
    learning_rate=opt.lr
    optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

    loo=LeaveOneOut()

    total_training_loss=0.0
    total_valid_loss=0.0

    for train_idx,valid_idx in loo.split(train_set):
        split=DataSplit(train_set,train_idx,valid_idx)
        train_loader,valid_loader=split.get_data_loader(train_set.__len__()-1,1)

        training_loss=0.0

        for i,(inputs,labels) in enumerate(train_loader):
            inputs=Variable(inputs.cuda(opt.cuda_devices))
            inputs=inputs.view([inputs.size(0),1])
            labels=Variable(labels.cuda(opt.cuda_devices))
            labels=labels.view([labels.size(0),1])

            optimizer.zero_grad()
            outputs=model(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            training_loss+=loss.item()*inputs.size(0)
        
        training_loss=training_loss/(len(train_loader)*inputs.size(0))
        total_training_loss+=training_loss


        model.eval()

        for inputs,labels in valid_loader:
            inputs=inputs.cuda(opt.cuda_devices)
            labels=labels.cuda(opt.cuda_devices)
            inputs=inputs.view([1,1])
            labels=labels.view([1,1])

            outputs=model(inputs.cuda(opt.cuda_devices))
            loss=criterion(outputs,labels)

        total_valid_loss+=loss

    avg_training_loss=total_training_loss/train_set.__len__()
    avg_valid_loss=total_valid_loss/train_set.__len__()

    return model,avg_training_loss,avg_valid_loss



##########          Training          ##########
# Linear Regression usign Leave one out as cross validation method
def train(train_set,valid_mode):
    input_dim=1
    output_dim=1
    model=LinearRegressionModel(input_dim,output_dim)
    model=model.cuda(opt.cuda_devices)
    model.train()

    best_model_param=copy.deepcopy(model.state_dict())
    training_loss_values=[]
    valid_loss_values=[]
    min_loss=float("inf")

    num_epochs=opt.epoch

    for epoch in range(num_epochs):
        print(f'Epoch: {epoch+1}/{num_epochs}')
        print('-'*len(f'Epoch: {epoch+1}/{num_epochs}'))
        
        if valid_mode=="leave one out":
            model,avg_training_loss,avg_valid_loss=loo_training(train_set,model)
        elif valid_mode=="five fold":
            # model,total_training_loss,total_valid_loss=five_fold_training(train_set,model)
            model,avg_training_loss,avg_valid_loss=five_fold_training(train_set,model)

        print(f'training_loss: {avg_training_loss:.4f}\tvalid_loss: {avg_valid_loss:.4f}\n')
        training_loss_values.append(avg_training_loss)
        valid_loss_values.append(avg_valid_loss)

        if avg_valid_loss<min_loss:
            min_loss=avg_valid_loss
            min_training_loss=avg_training_loss
            best_model_param=copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_param)

    print(f'min training_loss: {min_training_loss:.4f}\tmin valid_loss: {min_loss:.4f}\n')
    
    return model,training_loss_values,valid_loss_values

def main():
    data_set=RegressionDataset()
    total_sz=data_set.__len__()
    test_sz=int(0.25*total_sz)
    train_set,test_set=random_split(data_set,[total_sz-test_sz,test_sz])

    # best_model,training_loss,valid_loss=train(train_set,"leave one out")
    # loss_visualization(training_loss,valid_loss,"linear-reg-loo-loss.jpg")
    # test(test_set,best_model)

    best_model,training_loss,valid_loss=train(train_set,"five fold")
    loss_visualization(training_loss,valid_loss,"linear-rg-kf-loss.jpg")
    test(test_set,best_model)           

    

    
if __name__=="__main__":
    main()