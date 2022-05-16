# %%
# import matplotlib
# %matplotlib inline 

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.utils as utils
import time
import datetime
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from tqdm import tqdm
import wandb

#%%
from utils import create_dictionaries, transform_index_to_letter, calc_edit_distance, plot_attention, to_csv
from utils import LETTER_LIST, letter2index, index2letter
from dataloader import LibriSamples, LibriSamplesTest
from encoder import Encoder 
from decoder import Decoder
from traintest import train, eval, test
#%%
wandb.login()

#%%
cuda = torch.cuda.is_available()

print(cuda, sys.version)

device = torch.device("cuda" if cuda else "cpu")
num_workers = 4 if cuda else 0
print("Cuda = "+str(cuda)+" with num_workers = "+str(num_workers))
np.random.seed(11785)
torch.manual_seed(11785)

#%%
batch_size = 64

root = './hw4p2_student_data/hw4p2_student_data'

train_data = LibriSamples(root, 'train')
val_data = LibriSamples(root, 'dev')
test_data = LibriSamplesTest(root, 'test_order.csv')

train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=LibriSamples.collate_fn,
                          shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=LibriSamples.collate_fn,
                        shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=LibriSamplesTest.collate_fn,
                         shuffle=False, drop_last=False, num_workers=4, pin_memory=True) 


print("Batch size: ", batch_size)
print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
print("Val dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))

torch.cuda.empty_cache()
#%%
class Seq2Seq(nn.Module):
    '''
    We train an end-to-end sequence to sequence model comprising of Encoder and Decoder.
    This is simply a wrapper "model" for your encoder and decoder.
    '''
    def __init__(self, input_dim, vocab_size, encoder_hidden_dim, decoder_hidden_dim, embed_dim, key_value_size=128):
        super(Seq2Seq,self).__init__()
        self.encoder = Encoder(input_dim=input_dim, encoder_hidden_dim=encoder_hidden_dim, key_value_size=key_value_size)
        self.decoder = Decoder(vocab_size=vocab_size, decoder_hidden_dim=decoder_hidden_dim, embed_dim=embed_dim, key_value_size=key_value_size)

    def forward(self, x, x_len, y=None, mode='train', random_rate=0.5):
        key, value, encoder_len = self.encoder(x, x_len)
        predictions, attentions = self.decoder(key, value, encoder_len, y=y, mode=mode, random_rate=random_rate)
        return predictions, attentions

#%%
model = Seq2Seq(input_dim=13, vocab_size=len(LETTER_LIST), encoder_hidden_dim=256, decoder_hidden_dim=512, embed_dim=256, key_value_size=128)

model = model.to(device)
print(model)
#%%
# TODO: Define your model and put it on the device here
# ...

torch.cuda.empty_cache()

n_epochs = 100
min_distance = 1e9
# optimizer = optim.Adam(model.parameters(), # fill this out)
# Make sure you understand the implication of setting reduction = 'none'
# criterion = nn.CrossEntropyLoss(reduction='none')

config = {
    'model': "Seq2Seq_Model_4-4",
    # 'batch_size': batch_size,
    'epochs': 100,
    'lr': 1e-3,
    'weight_decay': 5e-6,
    }

#%%
optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=5, threshold=0.01, verbose = True)
criterion = nn.CrossEntropyLoss(reduction='none')

# wandb.init(name='Seq2Seq_Model4.4', 
#           project="11-785 - HW4P2",
#           notes='adam lr 1e-3 scheduler no bn, lockeddropout, 3xpblstm numlayers = 1',
#           config=config)

# wandb.watch(model)

# checkpoint = torch.load('{}'.format('./models/best_model_3-1.pth'))
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
# epoch = checkpoint['epoch']

for epoch in range(n_epochs):
    training_loss = train(model, optimizer, train_loader, criterion, optimizer, epoch)
    val_distance = eval(model, optimizer, val_loader, epoch)
    scheduler.step(val_distance)

    if (val_distance < min_distance):
        min_distance = val_distance
        torch.save({ 
            'epoch': epoch, 
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(), 
            }, './models/best_model_4-4.pth')
        print('Min distance {}'.format(min_distance))
        print('Model saved at {}'.format('./models/best_model_4-4.pth'))

pred = test(model, test_loader)

to_csv('output/predictions.csv', pred)
# %%
'''
Debugging suggestions from Eason, a TA from previous semesters:

(1) Decrease your batch_size to 2 and print out the value and shape of all intermediate variables to check if they satisfy the expectation
(2) Be super careful about the LR, don't make it too high. Too large LR would lead to divergence and your attention plot will never make sense
(3) Make sure you have correctly handled the situation for time_step = 0 when teacher forcing

(1) is super important and is the most efficient way for debugging. 
'''
'''
Tips for passing A from B (from easy to hard):
** You need to implement all of these yourself without utilizing any library **
(1) Increase model capacity. E.g. increase num_layer of lstm
(2) LR and Teacher Forcing are also very important, you can tune them or their scheduler as well. Do NOT change lr or tf during the warm-up stage!
(3) Weight tying
(4) Locked Dropout - insert between the plstm layers
(5) Pre-training decoder or train an LM to help make predictions
(5) Pre-training decoder to speed up the convergence: 
    disable your encoder and only train the decoder like train a language model
(6) Better weight initialization technique
(7) Batch Norm between plstm. You definitely can try other positions as well
(8) Data Augmentation. Time-masking, frequency masking
(9) Weight smoothing (avg the last few epoch's weight)
(10) You can try CNN + Maxpooling (Avg). Some students replace the entire plstm blocks with it and some just combine them together.
(11) Beam Search
'''