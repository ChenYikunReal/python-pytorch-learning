from torch.autograd import Variable
import torch

'''
如果有一个单一的输入操作需要梯度，则它的输出也需要梯度。
相反，只有所有输入都不需要梯度，输出才不需要。
如果其中所有的变量都不需要梯度进行，后向计算不会在子图中执行。
'''

x = Variable(torch.randn(5, 5))
y = Variable(torch.randn(5, 5))
z = Variable(torch.randn(5, 5), requires_grad=True)

# False 因为x和y都不需要梯度
a = x + y
print(a.requires_grad)

# True 因为z需要梯度
b = a + z
print(b.requires_grad)
