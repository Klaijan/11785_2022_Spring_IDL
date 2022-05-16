import torch
import os
import numpy as np

class LibriSamples(torch.utils.data.Dataset):
    
    def __init__(self, data_path, partition= "train"):
        # TODO

        self.X_dir = os.path.join(data_path, partition, 'mfcc')
        self.Y_dir = os.path.join(data_path, partition, 'transcript')

        self.X_files = os.listdir(self.X_dir)
        self.Y_files = os.listdir(self.Y_dir)

        assert(len(self.X_files) == len(self.Y_files))

    def __len__(self):
        # TODO
        return len(self.X_files)

    def __getitem__(self, ind):

        # TODO

        X_path = self.X_dir + '/' + self.X_files[ind]
        Y_path = self.Y_dir + '/' + self.Y_files[ind]

        X = np.load(X_path) # TODO: Load the mfcc npy file at the specified index ind in the directory
        Y = np.load(Y_path) # TODO: Load the corresponding transcripts

        # label = [LETTER_LIST.index(yy) for yy in Y[1:-1]]
        label = [LETTER_LIST.index(yy) for yy in Y]

        X = torch.tensor(X)
        X = (X - X.mean(axis=0))/X.std(axis=0)
        Y = np.array(label)
        Yy = torch.tensor(Y).type(torch.LongTensor)
        return X, Yy

    def collate_fn(batch):
        batch_x = [x for x,y in batch]
        batch_y = [y[1:] for x,y in batch]

        batch_x_pad = pad_sequence(batch_x, batch_first=True) # pad the sequence with pad_sequence
        lengths_x = [len(x) for x in batch_x] # Get original lengths of the sequence before padding

        batch_y_pad = pad_sequence(batch_y, batch_first=True) # TODO: pad the sequence with pad_sequence (already imported)
        lengths_y = [len(y) for y in batch_y] # TODO: Get original lengths of the sequence before padding

        return batch_x_pad, torch.tensor(lengths_x), batch_y_pad, torch.tensor(lengths_y)

class LibriSamplesTest(torch.utils.data.Dataset):

    def __init__(self, data_path, test_order):

        self.X_dir = os.path.join(data_path, 'test', 'mfcc')
        self.X_files = os.listdir(self.X_dir)

        if test_order:
            self.X_files = list(pd.read_csv(os.path.join(data_path, 'test', test_order)).file)
    
    def __len__(self):
        # TODO
        return len(self.X_files)
    
    def __getitem__(self, ind):
        X_path = self.X_dir + '/' + self.X_files[ind]
        X = np.load(X_path) # TODO: Load the mfcc npy file at the specified index ind in the directory
        X = torch.Tensor(X)
        X = (X - X.mean(axis=0))/X.std(axis=0)
        return X
    
    def collate_fn(batch):
        batch_x = [x for x in batch]
        batch_x_pad = pad_sequence(batch_x, batch_first=True) # pad the sequence with pad_sequence
        lengths_x = [len(x) for x in batch] # Get original lengths of the sequence before padding

        return batch_x_pad, torch.tensor(lengths_x)
