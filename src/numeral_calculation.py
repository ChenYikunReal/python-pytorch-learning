import torch

uninitialized = torch.Tensor(3, 2)
rand_initialized = torch.rand(3, 2)
matrix_with_ones = torch.ones(3, 2)
matrix_with_zeros = torch.zeros(3, 2)

size = rand_initialized.size()
shape = rand_initialized.shape
# True
print(size == shape)
