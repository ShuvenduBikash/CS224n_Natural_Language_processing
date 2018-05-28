import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from text_loader import TextDataset

hidden_size = 100
n_layers = 3
batch_size = 1
n_epochs = 100
n_characters = 128


class CharRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(CharRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, input, hidden):
        embed = self.embedding(input.view(1, -1))
        embed = embed.view(1, 1, -1)
        output, hidden = self.gru(embed, hidden)
        output = self.linear(output.view(1, -1))
        return output, hidden
    
    def init_hidden(self):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        
        if torch.cuda.is_available():
            hidden.cuda()
        
        return Variable(hidden)
        
    
def string2index(string):
    tensor = [ord(c) for c in string]
    tensor = torch.LongTensor(tensor)

    if torch.cuda.is_available():
        tensor = tensor.cuda()

    return Variable(tensor)
    
    
def train(line):
    input = string2index(line[:-1])
    target = string2index(line[1:])
    
    hidden = decoder.init_hidden()
    decoder_in = input[0]
    loss = 0
    
    for c in range(len(input)):
        output, hidden = decoder(decoder_in, hidden)
        loss += criterion(output, target[c].view(1))
        decoder_in = output.max(1)[1]
        
    decoder.zero_grad()
    loss.backward()
    decoder_optimizer.step()
    
    return loss.data[0] / len(input)
    
    
if __name__=='__main__':
    decoder = CharRNN(n_characters, hidden_size, n_characters, n_layers)
    if torch.cuda.is_available():
        decoder.cuda()
        
    decoder_optimizer = torch.optim.Adam(decoder.parameters())
    criterion = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(dataset=TextDataset(), batch_size=batch_size, shuffle=True)
    
    for epoch in range(1, n_epochs+1):
        for i, (lines, _) in enumerate(train_loader):
            loss = train(lines[0])
            if i % 100 == 0:
                print('[(%d %d%%) loss: %.4f]' % (epoch, epoch / n_epochs * 100, loss))