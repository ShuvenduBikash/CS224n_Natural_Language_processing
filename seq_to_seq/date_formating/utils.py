from babel.dates import format_date
from faker import Faker
import random
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader


fake = Faker()
fake.seed(12345)
random.seed(12345)


# Define format of the data we would like to generate
FORMATS = ['short',
           'medium',
           'long',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'd MMM YYY',
           'd MMMM YYY',
           'dd MMM YYY',
           'd MMM, YYY',
           'd MMMM, YYY',
           'dd, MMM YYY',
           'd MM YY',
           'd MMMM YYY',
           'MMMM d YYY',
           'MMMM d, YYY',
           'dd.MM.YY']

# change this if you want it to work with another language
LOCALES = ['en_US']


def load_date():
    """
    Load some fake dates
    :returns: tuple containing human readable string, machine readable string and date object
    """
    dt = fake.date_object()

    try:
        human_readable = format_date(dt, format=random.choice(FORMATS), locale='en_US')
        human_readable = human_readable.lower()
        human_readable = human_readable.replace(',', '')
        machine_readable = dt.isoformat()

    except AttributeError as e:
        return None, None, None

    return human_readable, machine_readable, dt


def load_dataset(m):
    """
    Load dataset with m examples and vocabularies
    m : number of example to generate
    returns: dataset, key value pairs for human, machine, inverted
    """
    human_vocab = set()
    machine_vocab = set()
    dataset = []
    Tx = 30
    
    for i in tqdm(range(m)):
        h, m, _ = load_date()
        if h is not None:
            dataset.append((h, m))
            human_vocab.update(tuple(h))
            machine_vocab.update(tuple(m))
    
    human = dict(zip(sorted(human_vocab) + ['<unk>', '<pad>'], 
                     list(range(len(human_vocab) + 2))))
    inv_machine = dict(enumerate(sorted(machine_vocab)))
    machine = {v:k for k,v in inv_machine.items()}
    inv_human = {v:k for k,v in human.items()}
 
    return dataset, human, machine, inv_human, inv_machine


def preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty):
    
    X, Y = zip(*dataset)
    
    X = np.array([string_to_int(i, Tx, human_vocab) for i in X])
    Y = np.array([string_to_int(t, Ty, machine_vocab) for t in Y])

    return X, Y


def string_to_int(string, length, vocab):
    """
    Converts all strings in the vocabulary into a list of integers representing the positions of the
    input string's characters in the "vocab"
    
    Arguments:
    string -- input string, e.g. 'Wed 10 Jul 2007'
    length -- the number of time steps you'd like, determines if the output will be padded or cut
    vocab -- vocabulary, dictionary used to index every character of your "string"
    
    Returns:
    rep -- list of integers (or '<unk>') (size = length) representing the position of the string's character in the vocabulary
    """
    
    #make lower to standardize
    string = string.lower()
    string = string.replace(',','')
    
    # trunk the string if it is larger then maxlen
    if len(string) > length:
        string = string[:length]
        
    rep = list(map(lambda x: vocab.get(x, '<unk>'), string))
    
    # pad the sting if it is less than maxlen
    if len(string) < length:
        rep += [vocab['<pad>']] * (length - len(string))
    
    #print (rep)
    return rep


class DateDataset(Dataset):
    
    def __init__(self, Tx, Ty, length=10000):
        self.length = length
        
        dataset, human_vocab, machine_vocab,inv_human_vocab, inv_machine_vocab = load_dataset(length)
        self.input_length = len(human_vocab)
        self.output_length = len(machine_vocab)
        self.inv_human_vocab = inv_human_vocab
        self.inv_machine_vocab = inv_machine_vocab
        self.X, self.Y = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)
        
            
    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.length


# Test the loader
if __name__ == "__main__":
    dataset = DateDataset(Tx=30, Ty=10, length=10)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=3,
                              shuffle=True)

    for i, (src, target) in enumerate(train_loader):
        print(i, "data  =>   ", src)   
        print(src.shape)