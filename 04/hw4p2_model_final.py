# %%
import os
import sys
import pandas as pd
import numpy as np
import Levenshtein as lev
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.utils as utils
import seaborn as sns
import matplotlib.pyplot as plt
import time
import random
import datetime
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from tqdm import tqdm
import wandb

#%%
# wandb.login()

#%%

cuda = torch.cuda.is_available()

print(cuda, sys.version)

device = torch.device("cuda" if cuda else "cpu")
num_workers = 4 if cuda else 0
print("Cuda = "+str(cuda)+" with num_workers = "+str(num_workers))
np.random.seed(11785)
torch.manual_seed(11785)

# The labels of the dataset contain letters in LETTER_LIST.
# You should use this to convert the letters to the corresponding indices
# and train your model with numerical labels.
LETTER_LIST = ['<sos>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
         'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', "'", ' ', '<eos>']

#%%
def create_dictionaries(letter_list):
    '''
    Create dictionaries for letter2index and index2letter transformations
    based on LETTER_LIST

    Args:
        letter_list: LETTER_LIST

    Return:
        letter2index: Dictionary mapping from letters to indices
        index2letter: Dictionary mapping from indices to letters
    '''
    letter2index = dict()
    index2letter = dict()
    # TODO
    for i, l in enumerate(letter_list):
        letter2index[l] = i
        index2letter[i] = l
    return letter2index, index2letter
    

def transform_index_to_letter(batch_indices):
    '''
    Transforms numerical index input to string output by converting each index 
    to its corresponding letter from LETTER_LIST

    Args:
        batch_indices: List of indices from LETTER_LIST with the shape of (N, )
    
    Return:
        transcripts: List of converted string transcripts. This would be a list with a length of N
    '''
    transcripts = []
    for batch in batch_indices:
        string_output = ''
        for idx in batch:
            letter = index2letter[idx]
            if letter == '<eos>':
                break
            else:
                string_output += letter
        transcripts.append(string_output)
    return transcripts
        
# Create the letter2index and index2letter dictionary
letter2index, index2letter = create_dictionaries(LETTER_LIST)

#%%
def calc_edit_distance(batch_text_1, batch_text_2, is_print):
    res = 0.0
    # import pdb; pdb.set_trace()
    for i, j in zip(batch_text_1, batch_text_2):
        if is_print:
            print('='*20)
            print('prediction')
            print(i)
            print('-'*10)
            print('target')
            print(j)
        distance = lev.distance(i, j)
        res += distance
    return res 

#%%

class LibriSamples(torch.utils.data.Dataset):
    
    def __init__(self, data_path, partition= "train"):
        self.X_dir = os.path.join(data_path, partition, 'mfcc')
        self.Y_dir = os.path.join(data_path, partition, 'transcript')

        self.X_files = os.listdir(self.X_dir)
        self.Y_files = os.listdir(self.Y_dir)

        assert(len(self.X_files) == len(self.Y_files))

    def __len__(self):
        return len(self.X_files)

    def __getitem__(self, ind):

        X_path = self.X_dir + '/' + self.X_files[ind]
        Y_path = self.Y_dir + '/' + self.Y_files[ind]

        X = np.load(X_path) # TODO: Load the mfcc npy file at the specified index ind in the directory
        Y = np.load(Y_path) # TODO: Load the corresponding transcripts

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
#%%
# test code for checking shapes
for data in val_loader:
    x, y, lx, ly = data
    print(x.shape, y.shape, lx.shape, len(ly))
    print(ly) # desired 
    break

#%%
class pBLSTM(nn.Module):
    '''
    Pyramidal BiLSTM
    Read paper and understand the concepts and then write your implementation here.

    At each step,
    1. Pad your input if it is packed
    2. Truncate the input length dimension by concatenating feature dimension
        (i) How should  you deal with odd/even length input? 
        (ii) How should you deal with input length array (x_lens) after truncating the input?
    3. Pack your input
    4. Pass it into LSTM layer

    To make our implementation modular, we pass 1 layer at a time.
    '''
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        self.blstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bias=True, batch_first=True, bidirectional=True)

    def forward(self, x):
        padded_x, lx = pad_packed_sequence(x, batch_first=True)
        lx = torch.LongTensor(lx)

        # chop off extra odd/even sequence
        padded_x = padded_x[:, :(padded_x.size(1) // 2) * 2, :] # (B, T, dim)

        # reshape to (B, T/2, dim*2)
        x_reshaped = padded_x.reshape(padded_x.size(0), padded_x.size(1) // 2, padded_x.size(2) * 2)
        lx = torch.div(lx, 2, rounding_mode='trunc')
        packed_x = pack_padded_sequence(x_reshaped, lengths=lx, batch_first=True, enforce_sorted=False)
        out, _ = self.blstm(packed_x)
        return out

#%%
# https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/lock_dropout.html

class LockedDropout(nn.Module):

    def __init__(self, p=0.3):
        self.p = p
        super().__init__()

    def forward(self, x):
        """
        Args:
            x (:class:`torch.FloatTensor` [sequence length, batch size, rnn hidden size]): Input to
                apply dropout too.
        """
        if not self.training or not self.p:
            return x
        
        x, lx = pad_packed_sequence(x, batch_first=True)
        lx = torch.LongTensor(lx)

        x = x.permute(1,0,2)

        x = x.clone()
        mask = x.new_empty(1, x.size(1), x.size(2), requires_grad=False).bernoulli_(1 - self.p)
        mask = mask.div_(1 - self.p)
        mask = mask.expand_as(x)

        mask = mask.permute(1,0,2) # (batch, seq_len, n_feats)
        x = x.permute(1,0,2) # (batch, seq_len, n_feats)

        out = pack_padded_sequence(x*mask, lengths=lx, batch_first=True, enforce_sorted=False)

        return out

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'p=' + str(self.p) + ')'

#%%
class Batchnorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.batchnorm = nn.BatchNorm1d(num_features=num_features)

    def forward(self, x):
        x, lx = pad_packed_sequence(x, batch_first=True)
        lx = torch.LongTensor(lx)

        x = x.permute(0,2,1) # (batch, n_feats, seq_len)
        x = self.batchnorm(x)

        x = x.permute(0,2,1) # (batch, seq_len, n_feats)
        x = pack_padded_sequence(x, lengths=lx, batch_first=True, enforce_sorted=False)

        return x
#%%
class Encoder(nn.Module):
    '''
    Encoder takes the utterances as inputs and returns the key, value and unpacked_x_len.

    '''
    def __init__(self, input_dim, encoder_hidden_dim, key_value_size=128):
        super(Encoder, self).__init__()
        # The first LSTM layer at the bottom
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=encoder_hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

        # Define the blocks of pBLSTMs
        # Dimensions should be chosen carefully
        # Hint: Bidirectionality, truncation...
        self.pBLSTMs = nn.Sequential(
            LockedDropout(),
            pBLSTM(input_dim=4*encoder_hidden_dim, hidden_dim=encoder_hidden_dim),
            pBLSTM(input_dim=4*encoder_hidden_dim, hidden_dim=encoder_hidden_dim),
            pBLSTM(input_dim=4*encoder_hidden_dim, hidden_dim=encoder_hidden_dim),
            LockedDropout()
        )
         
        # The linear transformations for producing Key and Value for attention
        # Hint: Dimensions when bidirectional lstm? 
        self.key_network = nn.Linear(2*encoder_hidden_dim, key_value_size) # in_features, out_features
        self.value_network = nn.Linear(2*encoder_hidden_dim, key_value_size)

    def forward(self, x, lx):
        """
        1. Pack your input and pass it through the first LSTM layer (no truncation)
        2. Pass it through the pyramidal LSTM layer
        3. Pad your input back to (B, T, *) or (T, B, *) shape
        4. Output Key, Value, and truncated input lens

        Key and value could be
            (i) Concatenated hidden vectors from all time steps (key == value).
            (ii) Linear projections of the output from the last pBLSTM network.
                If you choose this way, you can use the final output of
                your pBLSTM network.
        """

        # Pack input
        packed_input = pack_padded_sequence(x, lx, batch_first=True, enforce_sorted=False)
        # Pass to first lstm layer
        output, (h_n, c_n) = self.lstm(packed_input)
        # Pass to pyramid lstm layer
        output = self.pBLSTMs(output)
        out, lengths = pad_packed_sequence(output, batch_first=True)

        key = self.key_network(out)
        value = self.value_network(out)

        return key, value, lengths

#%%
encoder = Encoder(input_dim=40, encoder_hidden_dim=512)
# Try out your encoder on a tiny input before moving to the next step...
print(encoder)

torch.cuda.empty_cache()

#%%
def plot_attention(attention, outpath):
    # utility function for debugging
    plt.clf()
    fig = sns.heatmap(attention, cmap='GnBu')
    plt.show()
    plt.savefig(outpath)

class Attention(nn.Module):
    '''
    Attention is calculated using key and value from encoder and query from decoder.
    Here are different ways to compute attention and context:
    1. Dot-product attention
        energy = bmm(key, query) 
        # Optional: Scaled dot-product by normalizing with sqrt key dimension
        # Check "attention is all you need" Section 3.2.1
    * 1st way is what most TAs are comfortable with, but if you want to explore...
    2. Cosine attention
        energy = cosine(query, key) # almost the same as dot-product xD 
    3. Bi-linear attention
        W = Linear transformation (learnable parameter): d_k -> d_q
        energy = bmm(key @ W, query)
    4. Multi-layer perceptron
        # Check "Neural Machine Translation and Sequence-to-sequence Models: A Tutorial" Section 8.4
    
    After obtaining unnormalized attention weights (energy), compute and return attention and context, i.e.,
    energy = mask(energy) # mask out padded elements with big negative number (e.g. -1e9)
    attention = softmax(energy)
    context = bmm(attention, value)

    5. Multi-Head Attention
        # Check "attention is all you need" Section 3.2.2
        h = Number of heads
        W_Q, W_K, W_V: Weight matrix for Q, K, V (h of them in total)
        W_O: d_v -> d_v

        Reshape K: (B, T, d_k)
        to (B, T, h, d_k // h) and transpose to (B, h, T, d_k // h)
        Reshape V: (B, T, d_v)
        to (B, T, h, d_v // h) and transpose to (B, h, T, d_v // h)
        Reshape Q: (B, d_q)
        to (B, h, d_q // h)

        energy = Q @ K^T
        energy = mask(energy)
        attention = softmax(energy)
        multi_head = attention @ V
        multi_head = multi_head reshaped to (B, d_v)
        context = multi_head @ W_O
    '''
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, mask):
        """
        input:
            key: (batch_size, seq_len, d_k)
            value: (batch_size, seq_len, d_v)
            query: (batch_size, d_q)
        * Hint: d_k == d_v == d_q is often true if you use linear projections
        return:
            context: (batch_size, key_val_dim)
        
        """

        '''
        1. out = matmul q,k
        2. out = scale(out)
        3. out = mask(out)
        4. out = softmax(out)
        5. out = matmul(out, v)
        '''
        energy = torch.bmm(key, query.unsqueeze(2)).squeeze(2) # (N, T_max, key_size) * (N, context_size, 1) = (N, T_max, 1) -> (N, T_max)
        energy.masked_fill_(mask, -1e9) # mask out padded elements with big negative number (e.g. -1e9)
        attention = F.softmax(energy, dim=1)
        context = torch.bmm(attention.unsqueeze(1), value).squeeze(1) # (batch_size, key_val_dim)
        return context, attention
        # we return attention weights for plotting (for debugging)
#%%
class Decoder(nn.Module):
    '''
    As mentioned in a previous recitation, each forward call of decoder deals with just one time step.
    Thus we use LSTMCell instead of LSTM here.
    The output from the last LSTMCell can be used as a query for calculating attention.
    Methods like Gumble noise and teacher forcing can also be incorporated for improving the performance.
    '''
    def __init__(self, vocab_size, decoder_hidden_dim, embed_dim, key_value_size=128):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        # The number of cells is defined based on the paper
        self.lstm1 = nn.LSTMCell(input_size=embed_dim+key_value_size, hidden_size=decoder_hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size=decoder_hidden_dim, hidden_size=decoder_hidden_dim)
        self.lstm3 = nn.LSTMCell(input_size=decoder_hidden_dim, hidden_size=key_value_size)
    
        self.attention = Attention()     
        self.vocab_size = vocab_size

        self.character_prob = nn.Linear(2*key_value_size, vocab_size) # fill this out) #: d_v -> vocab_size
        self.key_value_size = key_value_size
        
        # Weight tying
        self.character_prob.weight = self.embedding.weight
        # self.init_weights()

    def forward(self, key, value, encoder_len, y=None, mode='train', random_rate=0.5):
        '''
        Args:
            key :(B, T, d_k) - Output of the Encoder (possibly from the Key projection layer)
            value: (B, T, d_v) - Output of the Encoder (possibly from the Value projection layer)
            y: (B, text_len) - Batch input of text with text_length
            mode: Train or eval mode for teacher forcing
        Return:
            predictions: the character perdiction probability 
        '''

        B, key_seq_max_len, key_value_size = key.shape # B = batch size

        if mode == 'train':
            max_len =  y.shape[1]
            char_embeddings = self.embedding(y)
        else:
            max_len = 600

        # TODO: Create the attention mask here (outside the for loop rather than inside) to aviod repetition
        mask = torch.arange(key_seq_max_len).unsqueeze(0) >= encoder_len.unsqueeze(1) # fill this out
        mask = mask.to(device)
        
        predictions = []
        # This is the first input to the decoder
        # What should the fill_value be? -> 0?
        prediction = torch.full((B,1), fill_value=0, device=device)
        hidden_states = [None, None, None] 
        
        # TODO: Initialize the context
        context = value[:, 0, :]

        attention_plot = [] # this is for debugging

        for i in range(max_len):
            if mode == 'train':
                # TODO: Implement Teacher Forcing -> ground truth y(t) as input t+1
                teacher_forcing = True if random.random() < random_rate else False
                if teacher_forcing:
                    if i == 0:
                        # This is the first time step
                        # Hint: How did you initialize "prediction" variable above?
                        # start all with <sos>
                        start_char = torch.zeros(B, dtype=torch.long).fill_(letter2index['<sos>']).to(device)
                        char_embed = self.embedding(start_char)
                    else:
                        # Otherwise, feed the label of the **previous** time step
                        # ground truth
                        char_embed = char_embeddings[:, i-1, :]
                else: # not teacher forcing
                    # add gumble noise to prediction
                    if i == 0:
                        char_embed = self.embedding(prediction.argmax(dim=-1))
                    else:
                        prediction = F.gumbel_softmax(prediction.type(torch.DoubleTensor)).to(device)
                        char_embed = self.embedding(prediction.argmax(dim=-1)) # embedding of the previous prediction
            else: # not train
                if i == 0:
                    start_char = torch.zeros(B, dtype=torch.long).fill_(letter2index['<sos>']).to(device)
                    char_embed = self.embedding(start_char)
                else:
                    char_embed = self.embedding(prediction.argmax(dim=-1)) # embedding of the previous prediction


            # what vectors should be concatenated as a context?
            y_context = torch.cat([char_embed, context], dim=1)
            # context and hidden states of lstm 1 from the previous time step should be fed
            hidden_states[0] = self.lstm1(y_context, hidden_states[0]) # Input of shape batch × input dimension; A tuple of LSTM hidden states of shape batch x hidden dimensions.

            # hidden states of lstm1 and hidden states of lstm2 from the previous time step should be fed
            hidden_states[1] = self.lstm2(hidden_states[0][0], hidden_states[1]) # gives out hidden states
            hidden_states[2] = self.lstm3(hidden_states[1][0], hidden_states[2]) # gives out hidden states
            query = hidden_states[2][0] # fill this out
            
            # Compute attention from the output of the second LSTM Cell
            context, attention = self.attention(query, key, value, mask)
            # We store the first attention of this batch for debugging
            attention_plot.append(attention.detach().cpu())
            
            # What should be concatenated as the output context?
            output_context = torch.cat([query, context], dim=1)
            prediction = self.character_prob(output_context)
            # store predictions
            predictions.append(prediction.unsqueeze(1))
        
        # Concatenate the attention and predictions to return
        attentions = torch.stack(attention_plot, dim=0)
        predictions = torch.cat(predictions, dim=1)
        return predictions, attentions

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)
        nn.init.zeros_(self.character_prob.weight)
        nn.init.uniform_(self.character_prob.weight, -initrange, initrange)
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
# Training
#%%

def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    model.to(device)
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, position=0, leave=False, desc='Train')
    start_time = time.time()
    running_loss = 0
    running_purplex = 0.0
    mode = 'train'
    random_rate = max(1 - 0.1*(epoch//5),0.5)

    
    # 0) Iterate through your data loader
    for i, (x, x_len, y, y_len) in enumerate(train_loader):
        
        # 1) Send the inputs to the device
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # 2) Pass your inputs, and length of speech into the model.
        predictions, attentions = model(x, x_len, y, mode=mode, random_rate = random_rate)
        attentions = attentions.permute(1, 0, 2)
        
        # 3) Generate a mask based on target length. This is to mark padded elements
        # so that we can exclude them from computing loss.
        # Ensure that the mask is on the device and is the correct shape.

        y_len = y_len.clone().detach().to(device)
        max_len = torch.max(y_len)
        mask = (torch.arange(0, max_len).repeat(y_len.size(0), 1).to(device) < y_len.unsqueeze(1).expand(y_len.size(0), max_len)).int() # fill this out
        mask = mask.to(device)

        # 4) Make sure you have the correct shape of predictions when putting into criterion
        loss = criterion(predictions.view(-1, predictions.size(2)), y.view(-1))
        # Use the mask you defined above to compute the average loss
        masked_loss = torch.sum(loss * mask.view(-1)) / torch.sum(mask)

        # 5) backprop

        masked_loss.backward()
        
        # Optional: Gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 2)

        # When computing Levenshtein distance, make sure you truncate prediction/target

        optimizer.step()

        current_loss = masked_loss.item()
        current_purplex = torch.exp(masked_loss).item()
        running_loss += current_loss
        running_purplex += current_purplex

        #########################################

        # model_name = 'hw4p2_model4-4'
        
        # Optional: plot your attention for debugging
        # plot_attention(attentions)
        # if i == 0:
        #     plot_attention(attentions[0,:x_len[0],:x_len[0]], 'plots/attention_{}_train_loadertest_epoch_{}_batch_{}'.format(model_name, epoch, i))

        batch_bar.set_postfix(
            epoch=epoch,
            loss='{:.04f}'.format(float(running_loss / (i + 1))),
            lr = '{:.04f}'.format(float(optimizer.param_groups[0]['lr']))
        )

        batch_bar.update()

    batch_bar.close()
    
    end_time = time.time()
    print("Finished Epoch: {}\ttrain loss: {:.4f}\ttrain perplex: {:.4f}\ttime: {:.4f}".format(epoch,\
          running_loss/len(train_loader), running_purplex/len(train_loader), end_time - start_time))

    # wandb.log({'epoch': epoch,
    #     'train_loss': float(running_loss / len(train_loader)),
    #     'lr': float(optimizer.param_groups[0]['lr'])})

    return running_loss/len(train_loader)

def val(model, val_loader, epoch):
    with torch.no_grad():
        model.eval()
        model.to(device)
        batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Eval')
        preds = []
        start_time = time.time()
        running_loss = 0
        running_purplex = 0.0
        running_distance = 0
        num_seq = 0
        mode = 'val'
        
        for i, (x, lx, y, ly) in enumerate(val_loader):
            x, y = x.to(device), y.to(device)
            predictions, attentions = model(x, lx, y, mode=mode)
            attentions = attentions.permute(1, 0, 2)

            # import pdb; pdb.set_trace()

            pred_text = transform_index_to_letter(predictions.argmax(-1).detach().cpu().numpy())
            y_text = transform_index_to_letter(y.detach().cpu().numpy())

            running_distance += calc_edit_distance(pred_text, y_text, 1 if i==0 else 0) 
            num_seq += len(pred_text) # batch_size

            # model_name = 'hw4p2_model4-4'
            # if i == 0:
            #     plot_attention(attentions[0,:lx[0],:lx[0]], 'plots/attention_{}_val_loadertest_epoch_{}_batch_{}'.format(model_name, epoch, i))

            batch_bar.set_postfix(
                epoch=epoch,
                loss='{:.04f}'.format(float(running_loss / (i + 1))),
                lr = '{:.04f}'.format(float(optimizer.param_groups[0]['lr']))
            )

            batch_bar.update()

        batch_bar.close()

        end_time = time.time()

        # wandb.log({'epoch': epoch,
        #     'edit_dist': running_distance/num_seq})

        print("Finished Epoch: {}\tedit distance: {:.4f}\ttime: {:.4f}".format(epoch, running_distance/num_seq, end_time - start_time))
        
    return running_distance/num_seq

#%%
def test(model, test_loader):
    model.eval()
    model.to(device)
    batch_bar = tqdm(total=len(test_loader), dynamic_ncols=True, position=0, leave=False, desc='Test')
    preds = []
    mode = 'test'

    for i, (x, lx) in enumerate(test_loader):
        x = x.to(device)
        
        with torch.no_grad():
            predictions, _ = model(x, lx, mode=mode)

        pred_text = transform_index_to_letter(predictions.argmax(-1).detach().cpu().numpy())

        preds.extend(pred_text)
        batch_bar.update()

    batch_bar.close()

    return preds


#%%
torch.cuda.empty_cache()

n_epochs = 100
min_distance = 1e9
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
    training_loss = train(model, train_loader, criterion, optimizer, epoch)
    val_distance = val(model, val_loader, epoch)
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

# with open("hw4p2_s2s_model4-4_bestep.csv", "w+") as f:
#     f.write("id,predictions\n")
#     for i in range(len(pred)):
#         f.write("{},{}\n".format(i, pred[i]))
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
