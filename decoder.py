import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import random
from utils import LETTER_LIST, letter2index, index2letter

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
    def __init__(self): #, dropout=0.0, method='dot'):
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
        energy = torch.bmm(key, query.unsqueeze(2)).squeeze(2)
        energy.masked_fill_(mask, -1e9) 
        attention = F.softmax(energy, dim=1)
        context = torch.bmm(attention.unsqueeze(1), value).squeeze(1) 

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
        # Hint: Be careful with the padding_idx
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        # The number of cells is defined based on the paper
        self.lstm1 = nn.LSTMCell(input_size=embed_dim+key_value_size, hidden_size=decoder_hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size=decoder_hidden_dim, hidden_size=decoder_hidden_dim)
        self.lstm3 = nn.LSTMCell(input_size=decoder_hidden_dim, hidden_size=key_value_size)
    
        self.attention = Attention()     
        self.vocab_size = vocab_size

        self.character_prob = nn.Linear(2*key_value_size, vocab_size) 
        self.key_value_size = key_value_size
        
        # Weight tying
        self.character_prob.weight = self.embedding.weight

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

        prediction = torch.full((B,1), fill_value=0, device=device)
        hidden_states = [None, None, None] 
        
        context = value[:, 0, :]

        attention_plot = [] 
        # import pdb; pdb.set_trace()

        for i in range(max_len):
            if mode == 'train':
                teacher_forcing = True if random.random() < random_rate else False
                if teacher_forcing:
                    if i == 0:
                        start_char = torch.zeros(B, dtype=torch.long).fill_(letter2index['<sos>']).to(device)
                        char_embed = self.embedding(start_char)
                    else:
                        char_embed = char_embeddings[:, i-1, :]
                else: # not teacher forcing
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

            y_context = torch.cat([char_embed, context], dim=1)
            hidden_states[0] = self.lstm1(y_context, hidden_states[0]) # Input of shape batch × input dimension; A tuple of LSTM hidden states of shape batch x hidden dimensions.

            hidden_states[1] = self.lstm2(hidden_states[0][0], hidden_states[1]) # gives out hidden states
            hidden_states[2] = self.lstm3(hidden_states[1][0], hidden_states[2]) # gives out hidden states
            query = hidden_states[2][0]
            
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