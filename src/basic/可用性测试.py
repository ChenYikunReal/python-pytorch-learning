import torch
from torch.backends import cudnn

a = torch.Tensor([1.])
# a = torch.Tensor([1.])
print(a.cuda())
# True
print(cudnn.is_acceptable(a.cuda()))