import math
import torch
import torch.nn as nn
import numpy as np

class LinearRegressionModel(nn.Module):
    def __init__(self,in_features,out_features):
        super(LinearRegressionModel,self).__init__()

        self.in_features=in_features
        self.out_features=out_features

        weight=np.random.normal(-1,1,(out_features,in_features))
        self.weight=torch.FloatTensor(weight)
        bias=np.random.normal(-1,1,out_features)
        self.bias=torch.FloatTensor(bias)


    def forward(self,x):
        out=self.x*self.weight
        out=sum(out)+self.bias
        return out

    def gradient(self,inputs,labels):
        Q,R=torch.qr(inputs)
        pseudo_inv=R.pinverse().mm(Q.t())

        self.weight=torch.mm(pseudo_inv,labels)


    """def __init__(self,in_features,out_features):
        super().__init__()
        self.poly = nn.Linear(in_features, out_features)
 
    def forward(self, input):
        out = self.poly(input)
        return out"""