from torch.autograd import Variable
import torch
import torchvision

'''
纯粹的inference模式下推荐使用volatile，当你确定你甚至不会调用.backward()时。
它比任何其他自动求导的设置更有效——它将使用绝对最小的内存来评估模型。
volatile也决定了require_grad is False。
'''

regular_input = Variable(torch.randn(5, 5))
volatile_input = Variable(torch.randn(5, 5), volatile=True)
model = torchvision.models.resnet18(pretrained=True)

# True
print(model(regular_input).requires_grad)

# False
print(model(volatile_input).requires_grad)

# True
print(model(volatile_input).volatile)

# True
print(model(volatile_input).creator is None)
