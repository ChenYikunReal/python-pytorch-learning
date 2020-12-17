import random
from torch.utils.tensorboard import SummaryWriter

value = 10
writer = SummaryWriter()
# writer.add_scalar('example', 3)
writer.add_scalar('游走图', value, 0)
for i in range(1, 10000):
    value += random.random()-0.5
    writer.add_scalar('游走图', value, i)
