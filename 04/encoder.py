import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

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
            # Batchnorm(num_features=2*encoder_hidden_dim),
            LockedDropout(),
            pBLSTM(input_dim=4*encoder_hidden_dim, hidden_dim=encoder_hidden_dim),
            # LockedDropout(),
            # Batchnorm(num_features=2*encoder_hidden_dim),
            pBLSTM(input_dim=4*encoder_hidden_dim, hidden_dim=encoder_hidden_dim),
            # Batchnorm(num_features=2*encoder_hidden_dim),
            # LockedDropout(),
            pBLSTM(input_dim=4*encoder_hidden_dim, hidden_dim=encoder_hidden_dim),
            # Batchnorm(num_features=2*encoder_hidden_dim)
            # Optional: dropout
            LockedDropout()
            # ...
            # nn.BatchNorm1D(encoder_hidden_dim)
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
