import torch
import torch.nn as nn
from torch.autograd import Variable

MAX_LENGTH = 100

SOS_token = chr(0)
EOS_token = 1


def cuda_variable(tensor):
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)


# string to char tensor
def str2tensor(msg, eos=False):
    tensor = [ord(c) for c in msg]
    if eos:
        tensor.append(EOS_token)

    return cuda_variable(torch.LongTensor(tensor))


class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers)

    def forward(self, word_inputs, hidden):
        seq_len = len(word_inputs)
        # formate shape
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)  # gru input shape = (seq_len, batch_dim, feature)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        # (num_layers * num_directions, batch, hidden_size)
        return cuda_variable(torch.zeros(self.n_layers, 1, self.hidden_size))



class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])

        return output, hidden

    def init_hidden(self):
        return cuda_variable(torch.zeros(self.n_layers, 1, hidden_size))
