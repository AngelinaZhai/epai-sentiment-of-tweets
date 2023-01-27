""" EPAI Class Models

This file contains the class objects for RNN, BiRNN, LSTM, BiLSTM and GRU.
"""
import gensim.downloader
import torch
from torch import nn
import torchtext

# global variables
GLOVE = torchtext.vocab.GloVe(name="6B", dim=50, max_vectors=10000)  # use 10k most common words
WORD2VEC = gensim.downloader.load("word2vec-google-news-300")


class Tweet_RNN(nn.Module):
    """
    The class object for the RNN.

    Attributes:
    emb: the type of embedding
    hidden_size: the number of layers
    nn: the actual neural network
    fc: the activation layer
    """

    # Tweet_RNN.__init__(self, input_size, hidden_size, num_classes)
    # param: self:Tweet_RNN
    # param: input_size:int
    # param: hidden_size:int
    # param: num_classes:int
    # param: embedding:str
    #    the string should be either: "glove", "word2vec" or "none"
    #    and correspond to the desired embedding
    # return: void
    # initializes the RNN
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, embedding: str) -> None:
        super(Tweet_RNN, self).__init__()
        if embedding == "glove":
            self.emb = nn.Embedding.from_pretrained(GLOVE.vectors)
        elif embedding == "word2vec":
            self.emb = nn.Embedding.from_pretrained(WORD2VEC)
        else:
            self.emb = nn.Embedding(input_size, num_classes)
        self.hidden_size = hidden_size
        self.nn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Sigmoid()

    # forward(self, x)
    # param: self:Tweet_RNN
    # param: x:torch.FloatTensor
    # initializes the RNN
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # Look up the embedding
        x = self.emb(x)
        # Set an initial hidden state
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        # Forward propagate the RNN
        out, _ = self.nn(x, h0)
        # Pass the output of the last time step to the classifier
        out = self.fc(out[:, -1, :])
        return out


class Tweet_BiRNN(nn.Module):
    """
    The class object for the BiRNN.

    Attributes:
    emb: the type of embedding
    hidden_size: the number of layers
    nn: the actual neural network
    fc: the activation layer
    """

    # Tweet_BiRNN.__init__(self, input_size, hidden_size, num_classes)
    # param: self:Tweet_BiRNN
    # param: input_size:int
    # param: hidden_size:int
    # param: num_classes:int
    # param: embedding:str
    #    the string should be either: "glove", "word2vec" or "none"
    #    and correspond to the desired embedding
    # return: void
    # initializes the BiRNN
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, embedding: str) -> None:
        super(Tweet_BiRNN, self).__init__()
        if embedding == "glove":
            self.emb = nn.Embedding.from_pretrained(GLOVE.vectors)
        elif embedding == "word2vec":
            self.emb = nn.Embedding.from_pretrained(WORD2VEC)
        else:
            self.emb = nn.Embedding(input_size, num_classes)
        self.hidden_size = hidden_size
        self.nn = nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Sigmoid()

    # forward(self, x)
    # param: self:Tweet_BiRNN
    # param: x:torch.FloatTensor
    # initializes the BiRNN
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # Look up the embedding
        x = self.emb(x)
        # Set an initial hidden state
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        # Forward propagate the RNN
        out, _ = self.nn(x, h0)
        # Pass the output of the last time step to the classifier
        out = self.fc(out[:, -1, :])
        return out


class Tweet_LSTM(nn.Module):
    """
    The class object for the LSTM.

    Attributes:
    emb: the type of embedding
    hidden_size: the number of layers
    nn: the actual neural network
    fc: the activation layer
    """

    # Tweet_LSTM.__init__(self, input_size, hidden_size, num_classes)
    # param: self:Tweet_LSTM
    # param: input_size:int
    # param: hidden_size:int
    # param: num_classes:int
    # param: embedding:str
    #    the string should be either: "glove", "word2vec" or "none"
    #    and correspond to the desired embedding
    # return: void
    # initializes the LSTM
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, embedding: str) -> None:
        super(Tweet_LSTM, self).__init__()
        if embedding == "glove":
            self.emb = nn.Embedding.from_pretrained(GLOVE.vectors)
        elif embedding == "word2vec":
            self.emb = nn.Embedding.from_pretrained(WORD2VEC)
        else:
            self.emb = nn.Embedding(input_size, num_classes)
        self.hidden_size = hidden_size
        self.nn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Sigmoid()

    # forward(self, x)
    # param: self:Tweet_LSTM
    # param: x:torch.FloatTensor
    # return: out:torch.FloatTensor
    # initializes the LSTM
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # Look up the embedding
        x = self.emb(x)
        # Set an initial hidden state and cell state
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        # Forward propagate the LSTM
        out, _ = self.nn(x, (h0, c0))
        # Pass the output of the last time step to the classifier
        out = self.fc(out[:, -1, :])
        return out


class Tweet_BiLSTM(nn.Module):
    """
    The class object for the BiLSTM.

    Attributes:
    emb: the type of embedding
    hidden_size: the number of layers
    nn: the actual neural network
    fc: the activation layer
    """

    # Tweet_BiLSTM.__init__(self, input_size, hidden_size, num_classes)
    # param: self:Tweet_BiLSTM
    # param: input_size:int
    # param: hidden_size:int
    # param: num_classes:int
    # param: embedding:str
    #    the string should be either: "glove", "word2vec" or "none"
    #    and correspond to the desired embedding
    # return: void
    # initializes the BiLSTM
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, embedding: str) -> None:
        super(Tweet_BiLSTM, self).__init__()
        if embedding == "glove":
            self.emb = nn.Embedding.from_pretrained(GLOVE.vectors)
        elif embedding == "word2vec":
            self.emb = nn.Embedding.from_pretrained(WORD2VEC)
        else:
            self.emb = nn.Embedding(input_size, num_classes)
        self.hidden_size = hidden_size
        self.nn = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Sigmoid()

    # forward(self, x)
    # param: self:Tweet_BiLSTM
    # param: x:torch.FloatTensor
    # return: out:torch.FloatTensor
    # initializes the BiLSTM
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # Look up the embedding
        x = self.emb(x)
        # Set an initial hidden state and cell state
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        # Forward propagate the BiLSTM
        out, _ = self.nn(x, (h0, c0))
        # Pass the output of the last time step to the classifier
        out = self.fc(out[:, -1, :])
        return out

class Tweet_GRU(nn.Module):
    """
    The class object for the GRU.

    Attributes:
    emb: the type of embedding
    hidden_size: the number of layers
    nn: the actual neural network
    fc: the activation layer
    """

    # Tweet_GRU.__init__(self, input_size, hidden_size, num_classes)
    # param: self:Tweet_GRU
    # param: input_size:int
    # param: hidden_size:int
    # param: num_classes:int
    # param: embedding:str
    #    the string should be either: "glove", "word2vec" or "none"
    #    and correspond to the desired embedding
    # return: void
    # initializes the GRU
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, embedding: str) -> None:
        super(Tweet_GRU, self).__init__()
        if embedding == "glove":
            self.emb = nn.Embedding.from_pretrained(GLOVE.vectors)
        elif embedding == "word2vec":
            self.emb = nn.Embedding.from_pretrained(WORD2VEC)
        else:
            self.emb = nn.Embedding(input_size, num_classes)
        self.hidden_size = hidden_size
        self.nn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Sigmoid()

    # forward(self, x)
    # param: self:Tweet_GRU
    # param: x:torch.FloatTensor
    # return: out:torch.FloatTensor
    # initializes the GRU
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # Look up the embedding
        x = self.emb(x)
        # Set an initial hidden state and cell state
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        # Forward propagate the GRU
        out, _ = self.nn(x, (h0, c0))
        # Pass the output of the last time step to the classifier
        out = self.fc(out[:, -1, :])
        return out
