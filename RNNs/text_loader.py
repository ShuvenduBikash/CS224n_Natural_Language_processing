import gzip
from torch.utils.data import Dataset, DataLoader
import os

# define root datadir
if os.path.exists('E:/Datasets'):
    root = 'E:/Datasets'
else:
    root = 'C:/Users/user/Documents/datasets'

    
class TextDataset(Dataset):
    
    def __init__(self, filename='shakespeare.txt.gz'):
        self.len = 0
        self.filename = root+'/NLP_data/'+filename
        
        with gzip.open(self.filename, 'rt') as f:
            self.targetLines = [x.strip() for x in f if x.strip()]
            self.srcLines = [x.lower().replace(' ', '')
                             for x in self.targetLines]
            self.len = len(self.srcLines)
            
    def __getitem__(self, index):
        return self.srcLines[index], self.targetLines[index]

    def __len__(self):
        return self.len


# Test the loader
if __name__ == "__main__":
    dataset = TextDataset()
    train_loader = DataLoader(dataset=dataset,
                              batch_size=3,
                              shuffle=True)

    for i, (src, target) in enumerate(train_loader):
        print(i, "data", src)    