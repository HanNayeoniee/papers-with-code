import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNNClassification(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        emb_size, 
        hidden_size, 
        num_layers=4):

        super(RNNClassification, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.RNN(
            input_size=emb_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)


    def forward(self, x, lens):
        #print(x)
        outs = self.emb(x)
        print("emb:", outs.size())
        outs_packed = pack_padded_sequence(outs, lens, batch_first=True, enforce_sorted=False)
        y, hidden = self.rnn(outs_packed)
        y, lens_unpacked = pad_packed_sequence(y, batch_first=True)
        print("rnn:", y.size())
        #y_last = y[:,-1]
        y = torch.stack([y[i,l-1, :] for i,l in zip(range(len(y)),lens)], dim=0)
        y = self.sigmoid(self.fc(y))
        return y

