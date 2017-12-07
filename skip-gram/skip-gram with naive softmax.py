import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import nltk
import random
import numpy as np
from collections import Counter
from .data_loader import load_training_data


class Skipgram(nn.Module):
    def __init__(self, vocab_size, projection_dim):
        pass


if __name__ == '__main__':
    EMBADDING_SIZE = 30
    BATCH_SIZE = 256
    EPOCH = 100

    losses = []
    model = Skipgram()

