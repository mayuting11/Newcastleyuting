import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
data=torch.FloatTensor([[1.0,2.0,3.0],[4.0,6.0,8.0]])

prob = F.softmax(data,dim=1) # dim = 0,在列上进行Softmax;dim=1,在行上进行Softmax
prob1=F.softmax(data,dim=0)
b=torch.tensor([[1,1],[2,3],[3,5]])
print(type(b))
actions = b.numpy()
print(type(actions))
