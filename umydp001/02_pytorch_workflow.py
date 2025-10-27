#The essence of machine learning and deep learning is to take some data from the past, 
# build an algorithm (like a neural network) to discover patterns in it 
# and use the discovered patterns to predict the future.

import torch
from torch import nn #nn stands for neural network and this package contains the building blocks for creating neural networks in PyTorch
import matplotlib.pyplot as plt # for visulaizing
#print(torch.__version__)

#Linear regression is also a type of machine learning algorithm, more specifically a supervised algorithm
weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
Y = weight * X + bias
print(X[:10])
print(Y[:10])