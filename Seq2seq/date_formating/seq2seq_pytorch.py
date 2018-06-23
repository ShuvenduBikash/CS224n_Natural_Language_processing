from utils import load_dataset, DateDataset
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

CUDA = torch.cuda.is_available()
# CUDA = False


def visualize_graph(model, data, comment):
    with SummaryWriter(comment=comment) as writer:
        writer.add_graph(model, data)


def cuda_variable(tensor):
    if CUDA:
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)


""" Define the models """


# define encoder
class EncoderRNN(nn.Module):
    """Encode the input sequence into a vector, which will be used to decode"""

    def __init__(self, input_size, hidden_size, num_layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers)

    def forward(self, word_inputs, hidden=None):
        """
        word_inputs: one batch of input data
                    (batch_size, seq_len) -> (64, 30)
        """
        input = word_inputs.transpose(0, 1).type(torch.LongTensor)  # Seqence first (30, 64)
        embedded = self.embedding(cuda_variable(input))  # (30, 64, 100)
        output, hidden = self.gru(embedded)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        """
        input: (seq_len, batch_size)
                (10, 64)
        """
        input = input.view(-1, BATCH_SIZE).type(torch.LongTensor)  # [10, 64]
        output = self.embedding(cuda_variable(input))  # [10, 64, 100]
        output, hidden_decoder = self.gru(output, hidden)  # [10, 64, 100]
        output = self.out(output)  # [10, 64, 11]
        return output, hidden_decoder


def translate():
    translate = ''
    src, _ = next(iter(train_loader))
    encoder_outputs, encoder_hidden = encoder(src)

    src = src[0]
    for i in range(len(src)):
        char = dataset.inv_human_vocab[src[i].item()]
        if char != '<pad>':
            translate += char
    translate += ' --> '

    hidden = encoder_hidden

    token = torch.zeros(1, BATCH_SIZE)
    for c in range(Ty):
        output, hidden = decoder(token, hidden)  # (1, 64, 11)
        token = output.cpu().data.numpy().argmax(2)  # (1, 64)
        translate += dataset.inv_machine_vocab[token[0, 0].item()]
        token = torch.from_numpy(token)

    print(translate)


if __name__ == '__main__':
    HIDDEN_SIZE = 100
    N_LAYERS = 1
    BATCH_SIZE = 64
    N_EPOCH = 100

    # Define the max length of input and output domain
    Tx = 30
    Ty = 10

    # load the dataset
    dataset = DateDataset(Tx, Ty, length=10000)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)  # (64, 30)
    sample_x, sample_y = next(iter(train_loader))

    # define input output shape
    INPUT_SIZE = dataset.input_length  # number of unique char in input    (37)
    OUTPUT_SIZE = dataset.output_length  # number of unique char in output   (11)

    encoder = EncoderRNN(INPUT_SIZE, HIDDEN_SIZE)
    decoder = DecoderRNN(HIDDEN_SIZE, OUTPUT_SIZE)

    if CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    print("Encoder : \n", encoder)
    print("Decoder : \n", decoder)

    # visualize_graph(encoder, sample_x, '_encoder')

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params)
    criterion = nn.CrossEntropyLoss()

    # Training
    print("Training for %d epochs" % N_EPOCH)
    for epoch in range(1, N_EPOCH + 1):
        for i, (src, target) in enumerate(train_loader):  # (64, 30) & (64, 10)

            if src.size()[0] != BATCH_SIZE:
                translate()
                continue

            encoder_outputs, encoder_hidden = encoder(src)
            target = cuda_variable(target.transpose(0, 1).type(torch.LongTensor))  # (10, 64)

            hidden = encoder_hidden
            loss = 0

            for c in range(Ty):
                token = target[c - 1].data if c else torch.zeros(1, BATCH_SIZE)
                output, hidden = decoder(token, hidden)
                loss += criterion(output[0], target[c])

            encoder.zero_grad()
            decoder.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.data[0] / Ty

            if i % 50 == 0:
                print('[(%d %d%%) %.4f]' %
                      (epoch, epoch / N_EPOCH * 100, loss))
