import math
import torch
import torch.nn as nn

class LinearRegressionModel(nn.Module):
    def __init__(self,in_features,out_features,bias=True):
        super(LinearRegressionModel,self).__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.bias=bias
        self.weight=nn.Parameter(torch.Tensor(out_features,in_features))

        if bias:
            self.bias=nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight,a=math.sqrt(5))

        if self.bias is not None:
            fan_in,_=nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound=1/math.sqrt(fan_in)
            nn.init.uniform_(self.bias,-bound,bound)
    
    def forward(self,x):
        # print("shape")
        # print(x.shape)
        _,dim=x.shape
        if dim!=self.in_features:
            print(f'Wrong input features. Please use tensor with {self.in_features} input features')
            return 0
        out=x.matmul(self.weight.t())

        if self.bias is not None:
            out+=self.bias
        
        return out