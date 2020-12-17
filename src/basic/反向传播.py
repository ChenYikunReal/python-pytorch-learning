from torch.autograd import Variable
import torch

'''
PyTorch不同于TensorFlow，TensorFlow是静态图，PyTorch是动态图。
PyTorch每一次向前传播（每一次运行代码）都会创建一个新的计算图。
'''

x = Variable(torch.randn(1, 10))
prev_h = Variable(torch.randn(1, 20))
w_h = Variable(torch.randn(20, 20))
w_x = Variable(torch.randn(20, 10))

i2h = torch.mm(w_x, x.t())
h2h = torch.mm(w_h, prev_h.t())
next_h = i2h + h2h
next_h = next_h.tanh()

next_h.backward(torch.ones(1, 20))
