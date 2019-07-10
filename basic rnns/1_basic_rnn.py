import torch
import torch.nn as nn
from torch.autograd import Variable

# Hand coded one hot encoding for "hello"
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

# One cell RNN input_dim = 4, output_dim=2, sequence -> 5
cell = nn.RNN(input_size=4, hidden_size=2, batch_first=True)

# provide random prev hidden-state (batch_size, sequence_size, feature)
hidden = Variable(torch.randn(1, 1, 2))

# Propagate through RNN
inputs = Variable(torch.tensor([h, e, l, l, o]).type(torch.FloatTensor))
for one in inputs:
    one = one.view(1, 1, -1)
    out, hidden = cell(one, hidden)
    print("One input size :", one.size(), "\tOut size: ", out.size())

# Adjustment - 1: We can do the whole at once through seq_len
inputs = inputs.view(1, 5, -1)
out, hidden = cell(inputs, hidden)
print("Squence input size : ", inputs.size(),"\tHidden size: ",hidden.size(), "\tOut size : ", out.size())

# Adjustment - 2: Now we want to feed batch of data
inputs = Variable(torch.Tensor([[h, e, l, l, o],
                                [e, o, l, l, l],
                                [l, l, e, e, l]]).type(torch.FloatTensor))
# update initial hidden size accrodingly
hidden = Variable(torch.randn(1, 3, 2))

out, hidden = cell(inputs, hidden)
print("Squence input size : ", inputs.size(),"\tHidden size: ",hidden.size(), "\tOut size : ", out.size())
