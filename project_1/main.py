import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,random_split,Subset
from sklearn.model_selection import LeaveOneOut,KFold
from pathlib import Path
from dataset import data_generator,RegressionDataset
from opt import parse_args


opt=parse_args()


def RMSE_loss(yhat,y):
    mse=((yhat - y) ** 2).sum() / yhat.data.nelement()
    return torch.sqrt(mse)


def linear_regression(x,weight):
    outputs=[]
    for i in x:
        out=i.view([1,len(i)]).mm(weight)
        outputs.append(out.numpy()) 
    return outputs


# Get the best weight using QR Decomposition
"""def get_best_weight(x,y,_lambda=0):
    Q,R=torch.qr(x)
    pseudo_inv=R.pinverse(_lambda).mm(Q.t())
    weight=torch.mm(pseudo_inv,y)
    return weight"""


def get_best_weight(x,y,_lambda=0):
    x_t=x.t()

    if _lambda==0:
        pseudo_inv=torch.inverse(x_t.mm(x))
    else:
        mat=x_t.mm(x)
        I=torch.eye(mat.size(0))
        pseudo_inv=torch.inverse(mat+_lambda*I)

    weight=torch.mm(pseudo_inv.mm(x_t),y)
    return weight

def get_poly_features(x,degree):
    poly_x=[]
    for each_x in x:
        poly_term=[each_x**n for n in range(1,degree+1)]
        poly_x.append(poly_term)
    return poly_x


def draw_fitting_plot(train_set,range,w_1,deg_1,legend,title,fname,w_2=None,deg_2=None,w_3=None,deg_3=None,w_4=None,deg_4=None):
    train_X=[]
    train_Y=[]

    for i,(inputs,labels) in enumerate(train_set):
        train_X.append(inputs)
        train_Y.append(labels)
    
    train_X=np.asarray(train_X)
    train_Y=np.asarray(train_Y)
    plt.scatter(train_X,train_Y,facecolors='none',edgecolors='b')

    X=np.linspace(range[0],range[1],100)
    
    deg_1_X=get_poly_features(X,deg_1)
    deg_1_X=np.hstack((deg_1_X,np.ones((len(deg_1_X),1))))
    deg_1_X=torch.FloatTensor(deg_1_X)
    w_1=torch.FloatTensor(w_1)
    deg_1_Y=linear_regression(deg_1_X,w_1)
    deg_1_Y=torch.FloatTensor(deg_1_Y)
    deg_1_Y=torch.flatten(deg_1_Y)
    plt.plot(X,deg_1_Y,'r',linewidth=0.5)

    if w_2 is not None:
        deg_2_X=get_poly_features(X,deg_2)
        deg_2_X=np.hstack((deg_2_X,np.ones((len(deg_2_X),1))))
        deg_2_X=torch.FloatTensor(deg_2_X)
        w_2=torch.FloatTensor(w_2)
        deg_2_Y=linear_regression(deg_2_X,w_2)
        deg_2_Y=torch.FloatTensor(deg_2_Y)
        deg_2_Y=torch.flatten(deg_2_Y)
        plt.plot(X,deg_2_Y,'g',linewidth=0.5)

    if w_3 is not None:
        deg_3_X=get_poly_features(X,deg_3)
        deg_3_X=np.hstack((deg_3_X,np.ones((len(deg_3_X),1))))
        deg_3_X=torch.FloatTensor(deg_3_X)
        w_3=torch.FloatTensor(w_3)
        deg_3_Y=linear_regression(deg_3_X,w_3)
        deg_3_Y=torch.FloatTensor(deg_3_Y)
        deg_3_Y=torch.flatten(deg_3_Y)
        plt.plot(X,deg_3_Y,'deepskyblue',linewidth=0.5)
    
    if w_4 is not None:
        deg_4_X=get_poly_features(X,deg_4)
        deg_4_X=np.hstack((deg_4_X,np.ones((len(deg_4_X),1))))
        deg_4_X=torch.FloatTensor(deg_4_X)
        w_4=torch.FloatTensor(w_4)
        deg_4_Y=linear_regression(deg_4_X,w_4)
        deg_4_Y=torch.FloatTensor(deg_4_Y)
        deg_4_Y=torch.flatten(deg_4_Y)
        plt.plot(X,deg_4_Y,'darkviolet',linewidth=0.5)
    
    plt.legend(legend,fontsize=16)
    plt.xlabel('x',fontsize=14)
    plt.ylabel('y',fontsize=14)
    plt.title(title,fontsize=14)
    plt.savefig(fname)
    plt.clf()
    

def test(test_set,degree,weight):
    test_X=[]
    test_Y=[]

    for i,(inputs,labels) in enumerate(test_set):
        test_X.append(inputs)
        test_Y.append(labels)

    test_X=get_poly_features(test_X,degree)
    test_X=np.hstack((test_X,np.ones((len(test_X),1))))
    test_X=torch.FloatTensor(test_X)
    test_Y=torch.FloatTensor(test_Y)
    test_Y=test_Y.view([test_Y.size(0),1])

    weight=torch.FloatTensor(weight)

    outputs=linear_regression(test_X,weight)
    outputs=torch.FloatTensor(outputs)
    outputs=outputs.view([outputs.size(0),1])

    loss=RMSE_loss(outputs,test_Y)
    avg_test_loss=loss/test_X.size(0)

    print(f'testing_loss: {avg_test_loss:.4f}\n')


def kf_train(dataset,degree=1,_lambda=0):
    kf=KFold(n_splits=5,random_state=None,shuffle=False)

    total_training_loss=0.0
    total_valid_loss=0.0
    total_weight=np.empty((degree+1,1))

    for train_idx,valid_idx in kf.split(dataset):
        train_set=Subset(dataset,train_idx)
        valid_set=Subset(dataset,valid_idx)

        train_X=[]
        train_Y=[]

        for i,(inputs,labels) in enumerate(train_set):
            train_X.append(inputs)
            train_Y.append(labels)

        train_X=get_poly_features(train_X,degree)
        train_X=np.hstack((train_X,np.ones((len(train_X),1))))
        train_X=torch.FloatTensor(train_X)
        train_Y=torch.FloatTensor(train_Y)
        train_Y=train_Y.view([train_Y.size(0),1])
            
        weight=get_best_weight(train_X,train_Y,_lambda)
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

        valid_X=get_poly_features(valid_X,degree)
        valid_X=np.hstack((valid_X,np.ones((len(valid_X),1))))
        valid_X=torch.FloatTensor(valid_X)
        valid_Y=torch.FloatTensor(valid_Y)
        valid_Y=valid_Y.view([valid_Y.size(0),1])

        outputs=linear_regression(valid_X,weight)
        outputs=torch.FloatTensor(outputs)

        loss=RMSE_loss(outputs,valid_Y)
        total_valid_loss+=loss/valid_X.size(0)

    best_weight=total_weight/dataset.__len__()
    avg_training_loss=total_training_loss/dataset.__len__()
    avg_valid_loss=total_valid_loss/dataset.__len__()

    print(f'training_loss: {avg_training_loss:.4f}\tvalid_loss: {avg_valid_loss:.4f}')

    return best_weight


def loo_train(dataset,degree=1,_lambda=0):
    loo=LeaveOneOut()

    total_training_loss=0.0
    total_valid_loss=0.0
    total_weight=np.zeros((degree+1,1))

    for train_idx,valid_idx in loo.split(dataset):
        train_set=Subset(dataset,train_idx)
        valid_set=Subset(dataset,valid_idx)

        train_X=[]
        train_Y=[]

        for i,(inputs,labels) in enumerate(train_set):
            train_X.append(inputs)
            train_Y.append(labels) 

        train_X=get_poly_features(train_X,degree)
        train_X=np.hstack((train_X,np.ones((len(train_X),1))))
        train_X=torch.FloatTensor(train_X)
        train_Y=torch.FloatTensor(train_Y)
        train_Y=train_Y.view([train_Y.size(0),1])

        weight=get_best_weight(train_X,train_Y,_lambda)
        total_weight+=weight.numpy()

        outputs=linear_regression(train_X,weight)
        outputs=torch.FloatTensor(outputs)
        outputs=outputs.view([outputs.size(0),1])
    
        loss=RMSE_loss(outputs,train_Y)
        total_training_loss+=loss/train_X.size(0)

        valid_X=[]
        valid_Y=[]

        for i,(inputs,labels) in enumerate(valid_set):
            valid_X.append(inputs)
            valid_Y.append(labels)

        valid_X=get_poly_features(valid_X,degree)
        valid_X=np.hstack((valid_X,np.ones((len(valid_X),1))))
        valid_X=torch.FloatTensor(valid_X)
        valid_Y=torch.FloatTensor(valid_Y)
        valid_Y=valid_Y.view([valid_Y.size(0),1])

        outputs=linear_regression(valid_X,weight)
        outputs=torch.FloatTensor(outputs)
        outputs=outputs.view([outputs.size(0),1])

        loss=RMSE_loss(outputs,valid_Y)
        total_valid_loss+=loss

    best_weight=total_weight/dataset.__len__()
    avg_training_loss=total_training_loss/dataset.__len__()
    avg_valid_loss=total_valid_loss/dataset.__len__()

    print(f'training_loss: {avg_training_loss:.4f}\tvalid_loss: {avg_valid_loss:.4f}')
    return best_weight
        

def main():
    if Path('./data/data.csv').is_file()==False:
        data_generator('linear','./data/data.csv',20)

    data_set=RegressionDataset('./data/data.csv')
    total_sz=data_set.__len__()
    test_sz=int(0.25*total_sz)
    train_set,test_set=random_split(data_set,[total_sz-test_sz,test_sz])

    origin_train_set=train_set
    origin_test_set=test_set

    print("Regression Linear Leave One Out Degree 1 20 Data Points")
    weight_loo_deg1=loo_train(train_set,1)
    test(test_set,1,weight_loo_deg1)

    print("Regression Linear Five Fold Degree 1 20 Data Points")
    weight_kf_deg1=kf_train(train_set,1)
    test(test_set,1,weight_kf_deg1)

    print("Regression Linear Leave One Out Degree 5 20 Data Points")
    weight_loo_deg5=loo_train(train_set,5)
    test(test_set,5,weight_loo_deg5)

    print("Regression Linear Five Fold Degree 5 20 Data Points")
    weight_kf_deg5=kf_train(train_set,5)
    test(test_set,5,weight_kf_deg5)

    print("Regression Linear Leave One Out Degree 10 20 Data Points")
    weight_loo_deg10=loo_train(train_set,10)
    test(test_set,10,weight_loo_deg10)

    print("Regression Linear Five Fold Degree 10 20 Data Points")
    weight_kf_deg10=kf_train(train_set,10)
    test(test_set,10,weight_kf_deg10)

    print("Regression Linear Leave One Out Degree 14 20 Data Points")
    weight_loo_deg14=loo_train(train_set,14)
    test(test_set,14,weight_loo_deg14)

    print("Regression Linear Five Fold Degree 14 20 Data Points")
    weight_kf_deg14=kf_train(train_set,14)
    test(test_set,14,weight_kf_deg14)

    draw_fitting_plot(train_set,[-3,3],weight_loo_deg1,1,['degree=1','degree=5','degree=10','degree=14'],'Linear Data Live One Out Curve','./figure/linear-loo.jpg',weight_loo_deg5,5,weight_loo_deg10,10,weight_loo_deg14,14)
    draw_fitting_plot(train_set,[-3,3],weight_kf_deg1,1,['degree=1','degree=5','degree=10','degree=14'],'Linear Data Five Folds Curve','./figure/linear-kf.jpg',weight_kf_deg5,5,weight_kf_deg10,10,weight_kf_deg14,14)

    if Path('./data/sine_data.csv').is_file()==False:
        data_generator('sin','./data/sin_data.csv',20)

    data_set=RegressionDataset('./data/sin_data.csv')
    total_sz=data_set.__len__()
    test_sz=int(0.25*total_sz)
    train_set,test_set=random_split(data_set,[total_sz-test_sz,test_sz])
    
    print('Regression Sine Leave One Out Degree 5 20 Data Points')
    weight_loo_deg5=loo_train(train_set,5)
    test(test_set,5,weight_loo_deg5)
    
    print('Regression Sine Five Fold Degree 5 20 Data Points')
    weight_kf_deg5=kf_train(train_set,5)
    test(test_set,5,weight_kf_deg5)

    print('Regression Sine Leave One Out Degree 10 20 Data Points')
    weight_loo_deg10=loo_train(train_set,10)
    test(test_set,10,weight_loo_deg10)
    
    print('Regression Sine Five Fold Degree 10 20 Data Points')
    weight_kf_deg10=kf_train(train_set,10)
    test(test_set,10,weight_kf_deg10)

    print('Regression Sine Leave One Out Degree 14 20 Data Points')
    weight_loo_deg14=loo_train(train_set,14)
    test(test_set,14,weight_loo_deg14)
    
    print('Regression Sine Five Fold Degree 14 20 Data Points')
    weight_kf_deg14=kf_train(train_set,14)
    test(test_set,14,weight_kf_deg14)

    draw_fitting_plot(train_set,[0,1],weight_loo_deg5,5,['degree=5','degree=10','degree=14'],'Sine Data Live One Out Curve','./figure/sine-loo.jpg',weight_loo_deg10,10,weight_loo_deg14,14)
    draw_fitting_plot(train_set,[0,1],weight_kf_deg5,5,['degree=5','degree=10','degree=14'],'Sine Data Five Folds Curve','./figure/sine-kf.jpg',weight_kf_deg10,10,weight_kf_deg14,14)

    if Path('./data/data_320.csv').is_file()==False:
        data_generator('linear','./data/data_320.csv',320)
    
    data_set=RegressionDataset('./data/data_320.csv')
    total_sz=data_set.__len__()
    delete_sz=260
    use_set,delete_set=random_split(data_set,[total_sz-delete_sz,delete_sz])

    total_sz=use_set.__len__()
    test_sz=int(0.25*total_sz)
    train_set,test_set=random_split(use_set,[total_sz-test_sz,test_sz])

    print('Regression Linear Leave One Out Degree 14 60 Data Points')
    weight_loo_data60=loo_train(train_set,14)
    test(test_set,14,weight_loo_data60)

    print('Regression Linear Five Fold Degree 14 60 Data Points')
    weight_kf_data60=kf_train(train_set,14)
    test(test_set,14,weight_kf_data60)

    total_sz=data_set.__len__()
    delete_sz=160
    use_set,delete_set=random_split(data_set,[total_sz-delete_sz,delete_sz])

    total_sz=use_set.__len__()
    test_sz=int(0.25*total_sz)
    train_set,test_set=random_split(use_set,[total_sz-test_sz,test_sz])

    print('Regression Linear Leave One Out Degree 14 160 Data Points')
    weight_loo_data160=loo_train(train_set,14)
    test(test_set,14,weight_loo_data160)

    print('Regression Linear Five Fold Degree 14 160 Data Points')
    weight_kf_data160=kf_train(train_set,14)
    test(test_set,14,weight_kf_data160)

    total_sz=data_set.__len__()
    delete_sz=0
    use_set,delete_set=random_split(data_set,[total_sz-delete_sz,delete_sz])

    total_sz=use_set.__len__()
    test_sz=int(0.25*total_sz)
    train_set,test_set=random_split(use_set,[total_sz-test_sz,test_sz])

    print('Regression Linear Leave One Out Degree 14 320 Data Points')
    weight_loo_data320=loo_train(train_set,14)
    test(test_set,14,weight_loo_data320)

    print('Regression Linear Five Fold Degree 14 320 Data Points')
    weight_kf_data320=kf_train(train_set,14)
    test(test_set,14,weight_kf_data320)

    draw_fitting_plot(origin_train_set,[-3,3],weight_loo_data60,14,['m=60','m=160','m=320'],'Linear Data Different m Live One Out Curve','./figure/data-m-loo.jpg',weight_loo_data160,14,weight_loo_data320,14)
    draw_fitting_plot(origin_train_set,[-3,3],weight_kf_data60,14,['m=60','m=160','m=320'],'Linear Data Different m Five Folds Curve','./figure/data-m-kf.jpg',weight_kf_data160,14,weight_kf_data320,14)

    print('Regularization 0.001/m Linear Five Fold Degree 14 20 Data Points')
    _lambda=0.001/20
    weight_kf_0001l=kf_train(origin_train_set,14,_lambda)
    test(origin_test_set,14,weight_kf_0001l)
    
    print('Regularization 1/m Linear Five Fold Degree 14 20 Data Points')
    _lambda=float(1)/20
    weight_kf_1l=kf_train(origin_train_set,14,_lambda)
    test(origin_test_set,14,weight_kf_1l)

    print('Regularization 1000/m Linear Five Fold Degree 14 20 Data Points')
    _lambda=float(1000)/20
    weight_kf_1000l=kf_train(origin_train_set,14,_lambda)
    test(origin_test_set,14,weight_kf_1000l)

    draw_fitting_plot(origin_train_set,[-3,3],weight_kf_0001l,14,['0.001/m','1/m','1000/m'],'Linear Data Five Fold with Regularization Curve','./figure/regularization-kf.jpg',weight_kf_1l,14,weight_kf_1000l,14)


if __name__=='__main__':
    main()