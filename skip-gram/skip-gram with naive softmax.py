import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import nltk
import random
import numpy as np
from collections import Counter

flatten = lambda l: [item for sublist in l for item in sublist]

USE_CUDA = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


# helper function
def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex:eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch

    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch


def prepare_sequence(seq, word2index):
    idxs = list(map(lambda w: word2index[w] if w in word2index.keys() else word2index["<UNK>"], seq))
    return Variable(LongTensor(idxs))


def prepare_word(word, word2index):
    return Variable(LongTensor([word2index[word]]) if word in word2index.keys() else LongTensor([word2index["<UNK>"]]))


# loading data
corpus = list(nltk.corpus.gutenberg.sents('melville-moby_dick.txt'))[:100]  # sampling sentences for test
corpus = [[word.lower() for word in sent] for sent in corpus]


# Extract stop words from unigram distributions tails
word_count = Counter(flatten(corpus))

# amount of words to be count as most and least common
border =int(len(word_count)*0.01) 

# getting the most common and list common
stopwords = word_count.most_common()[:border]+list(reversed(word_count.most_common()))[:border]
stopwords = [s[0] for s in stopwords]


'''Build Vocab'''
vocab = list(set(flatten(corpus))-set(stopwords))
vocab.append('<UNK>')

# build word to index
word2index = {'<UNK>' : 0} 

for vo in vocab:
    if vo not in word2index.keys():
        word2index[vo]=len(word2index)

index2word = {v:k for k,v in word2index.items()} 






