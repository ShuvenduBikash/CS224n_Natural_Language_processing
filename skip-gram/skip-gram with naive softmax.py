import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import nltk
import random
import numpy as np
from collections import Counter
from .data_loader import load_training_data, getBatch, prepare_sequence


class Skipgram(nn.Module):

    def __init__(self, vocab_size, projection_dim):
        super(Skipgram, self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, projection_dim)
        self.embedding_u = nn.Embedding(vocab_size, projection_dim)

        self.embedding_v.weight.data.uniform_(-1, 1)  # init
        self.embedding_u.weight.data.uniform_(0, 0)  # init
        # self.out = nn.Linear(projection_dim,vocab_size)

    def forward(self, center_words, target_words, outer_words):
        center_embeds = self.embedding_v(center_words)  # B x 1 x D
        target_embeds = self.embedding_u(target_words)  # B x 1 x D
        outer_embeds = self.embedding_u(outer_words)  # B x V x D

        scores = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)  # Bx1xD * BxDx1 => Bx1
        norm_scores = outer_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)  # BxVxD * BxDx1 => BxV

        nll = -torch.mean(
            torch.log(torch.exp(scores) / torch.sum(torch.exp(norm_scores), 1).unsqueeze(1)))  # log-softmax

        return nll  # negative log likelihood

    def prediction(self, inputs):
        embeds = self.embedding_v(inputs)

        return embeds


if __name__ == '__main__':
    EMBEDDING_SIZE = 30
    BATCH_SIZE = 256
    EPOCH = 100

    # loading the data
    train_data, word2index, index2word, vocab = load_training_data()

    # defining the model
    losses = []
    model = Skipgram(len(word2index), EMBEDDING_SIZE)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # training the mode
    for epoch in range(EPOCH):
        for i, batch in enumerate(getBatch(BATCH_SIZE, train_data)):
            inputs, targets = zip(*batch)

            inputs = torch.cat(inputs)  # B x 1
            targets = torch.cat(targets)  # B x 1
            vocabs = prepare_sequence(list(vocab), word2index).expand(inputs.size(0), len(vocab))  # B x V
            model.zero_grad()

            loss = model(inputs, targets, vocabs)

            loss.backward()
            optimizer.step()

            losses.append(loss.data.tolist()[0])

        if epoch % 10 == 0:
            print("Epoch : %d, mean_loss : %.02f" % (epoch, np.mean(losses)))
            losses = []
