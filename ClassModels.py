""" EPAI Class Models

This file contains the class objects for RNN, BiRNN, LSTM, BiLSTM and GRU.
"""

import torch
from torch import nn
import torchtext

# global variables
GLOVE = torchtext.vocab.GloVe(name="6B", dim=50, max_vectors=10000)  # use 10k most common words

class Tweet_RNN(nn.Module):
    """
    The class object for the RNN
    """

    def __init__(self, input_size, hidden_size, num_classes):
        super(Tweet_RNN, self).__init__()
        self.emb = nn.Embedding.from_pretrained(GLOVE.vectors)
        # add other embeddoing options
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Look up the embedding
        x = self.emb(x)
        # Set an initial hidden state
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        # Forward propagate the RNN
        out, _ = self.rnn(x, h0)
        # Pass the output of the last time step to the classifier
        out = self.fc(out[:, -1, :])
        return out


class Tweet_LSTM(nn.Module):
    """
    The class object for the LSTM
    """
    # ... ****CONT***
    # param: embedding:string
    #    the string should be either: "glove", "word2vec" or "none"
    # return: void
    # initializes the LSTM
    def __init__(self, input_size, hidden_size, num_classes, embedding):
        super(Tweet_LSTM, self).__init__()
        if embedding == "glove":
            self.emb = nn.Embedding.from_pretrained(GLOVE.vectors)
        elif embedding == "word2vec":
            # finish below line
            # self.emb = nn.Embedding.from_pretrained(torchtext.models.ROBERTA_LARGE_ENCODER)
        else:
            self.emb = nn.Embedding(input_size, num_classes) # ??
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Look up the embedding
        x = self.emb(x)
        # Set an initial hidden state and cell state
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        # Forward propagate the LSTM
        out, _ = self.rnn(x, (h0, c0))
        # Pass the output of the last time step to the classifier
        out = self.fc(out[:, -1, :])
        return out
