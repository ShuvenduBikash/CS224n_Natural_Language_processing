import os
from io import open
import torch
import pickle
import codecs


def save_obj(obj, name):
    """
    Saves a object into file (e.g. Dictionary)
    :param obj: (any) object e.g. dict
    :param name: (string) filename
    :return: None
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """
    Loads a saved object from file
    :param name: (str) filename_path
    :return: object
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class CorpusEN(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            # for i in range(100):
            for line in f:
                # line = f.read()
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids


class Corpus(object):
    def __init__(self, path):
        if os.path.exists(os.path.join(path, 'vocab.pkl')):
            self.dictionary = load_obj(os.path.join(path, 'vocab'))
        else:
            self.dictionary = Dictionary()
            self.dictionary.add_word('<eos>')
            self.dictionary.add_word(',')
            self.dictionary.add_word('?')
            self.dictionary.add_word(':')
            self.dictionary.add_word('।')
            self.dictionary.add_word('!')
            self.dictionary.add_word('-')
            with open(os.path.join(path, 'unigram_5lkh.txt'), 'r', encoding="utf8") as f:
                for line in f:
                    try:
                        w, _ = line.split(' ')
                        self.dictionary.add_word(w)
                    except:
                        pass
            save_obj(self.dictionary, os.path.join(path, 'vocab'))

        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        # with open(path, 'r', encoding="utf8") as f:
        #     for line in f:
        #         words = line.split() + ['<eos>']
        #         for word in words:
        #             self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            # for i in range(100):
            vocab = set(self.dictionary.idx2word)
            for line in f:
                # line = f.read()
                words = line.split() + ['<eos>']

                unknown = False
                for w in words:
                    if w not in vocab:
                        unknown = True

                if not unknown:
                    ids = []
                    for word in words:
                        ids.append(self.dictionary.word2idx[word])
                    idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids