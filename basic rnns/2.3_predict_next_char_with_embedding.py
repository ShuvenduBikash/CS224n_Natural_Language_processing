import torch
import torch.nn as nn
from torch.autograd import Variable

idx2char = ['h', 'i', 'e', 'l', 'o']
# Train data
x_data = [[0, 1, 0, 2, 3, 3]]   # hihell
y_data = [1, 0, 2, 3, 3, 4]    # ihello

inputs = Variable(torch.LongTensor(x_data))
labels = Variable(torch.LongTensor(y_data))

# define parameters
batch_size = 1
seq_len = 6
input_size = 5
num_classes = 5
embedding_dim = 10
hidden_size=10

# define the Model
class Model(nn.Module):

    def __init__(self, input_size, embedding_dim, num_classes):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        hidden = torch.zeros((1, batch_size, hidden_size))
        input = self.embedding(x)
        output, hidden = self.rnn(input, hidden)
        output = self.linear(output)
        return output.view(-1, num_classes)


# Instantiate model
model = Model(input_size, embedding_dim, num_classes)
print(model)

# deinfe criterion
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# train the Model
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(1)
    idx = idx.data.numpy()
    result_str = [idx2char[c] for c in idx.squeeze()]
    print("epoch: %d, loss: %1.3f" % (epoch + 1, loss.data[0]))
    print("Predicted string: ", ''.join(result_str))
