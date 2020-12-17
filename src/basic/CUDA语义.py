import torch

'''
torch.cuda会记录当前选择的GPU，并且分配的所有CUDA张量将在上面创建。
可以使用torch.cuda.device上下文管理器更改所选设备。
一旦张量被分配，您可以直接对其进行操作，而不考虑所选择的设备，结果将始终放在与张量相同的设备上。
'''

x = torch.cuda.FloatTensor(1)
# x.get_device() == 0
y = torch.FloatTensor(1).cuda()
# y.get_device() == 0

with torch.cuda.device(1):
    # allocates a tensor on GPU 1
    a = torch.cuda.FloatTensor(1)

    # transfers a tensor from CPU to GPU 1
    b = torch.FloatTensor(1).cuda()
    # a.get_device() == b.get_device() == 1

    c = a + b
    # c.get_device() == 1

    z = x + y
    # z.get_device() == 0

    # even within a context, you can give a GPU id to the .cuda call
    d = torch.randn(2).cuda(2)
    # d.get_device() == 2

'''
报错：
THCudaCheck FAIL file=..\torch\csrc\cuda\Module.cpp line=59 error=101 : invalid device ordinal
Traceback (most recent call last):
  File "D:/PyCharm/pytorch_learning/basic_test/CUDA语义.py", line 14, in <module>
    with torch.cuda.device(1):
  File "D:\Python\Anaconda\lib\site-packages\torch\cuda\__init__.py", line 209, in __enter__
    torch._C._cuda_setDevice(self.idx)
RuntimeError: cuda runtime error (101) : invalid device ordinal at ..\torch\csrc\cuda\Module.cpp:59
'''
