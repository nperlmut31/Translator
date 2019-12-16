import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.functional import cross_entropy
import torch.nn.functional as f
import numpy as np


class Encoder(nn.Module):

    def __init__(self, num_embeddings=5000, embedding_dim=100, hidden_size=100,
                 padding_idx = 2, relu=True):
        super().__init__()

        self.relu = relu

        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim,
                                      padding_idx=padding_idx)

        self.GRU = nn.GRU(input_size=embedding_dim,
                          hidden_size=hidden_size,
                          num_layers=1,
                          bidirectional=True,
                          batch_first=False)

    def forward(self, x):
        #input = [S, B, H]
        #output = [S, B, 2*H]
        #h = [2, B, H]
        x = self.embedding(x)
        if self.relu:
            x = nn.ReLU()(x)
        input = x.permute(1, 0, 2)
        output, h = self.GRU(input)
        return output, h



class Decoder(nn.Module):

    def __init__(self, num_embeddings=5000, embedding_dim=100, hidden_size=200, padding_idx=2, relu=True):
        super().__init__()

        self.relu = relu

        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim, padding_idx=padding_idx)

        self.GRU = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size,
                          num_layers=2, bidirectional=False, batch_first=False)

        self.concat_matrix = nn.Parameter(torch.rand(hidden_size + embedding_dim, embedding_dim, requires_grad=True))

        nn.init.normal_(self.concat_matrix)

        #Use GRU_1 if there is no feedback, use GRU_2 if there is feedback.
    def forward(self, *args):
        x, h = args[0], args[1]
        x = self.embedding(x)
        if self.relu:
            x = nn.ReLU()(x)
        x = x.permute(1, 0, 2)
        #x = [1, B, E]
        #h = [2, B, H]
        if len(args) <= 2:
            output, h = self.GRU(x, h)
            return output, h
        else:
            feedback = args[2]
            x_1 = torch.cat([x, feedback], dim=2)
            x_1 = torch.matmul(x_1, self.concat_matrix)
            output, h = self.GRU_2(x_1, h)
            return output, h


class Attention(nn.Module):

    def __init__(self, hidden_size = 100, D = 2, cuda=False):
        super().__init__()

        self.cuda = cuda
        self.D = D
        self.sigma = self.D/2

        self.W_a = nn.Parameter(torch.rand(hidden_size, hidden_size, requires_grad=True))
        self.W_p = nn.Parameter(torch.rand(hidden_size, hidden_size, requires_grad=True))
        self.v_p = nn.Parameter(torch.rand(hidden_size, 1, requires_grad=True))
        self.W_c = nn.Parameter(torch.rand(2*hidden_size, hidden_size, requires_grad=True))

        for p in self.named_parameters():
            nn.init.normal_(p[1])


    def alignment(self, e, d):
        #output = [S, B]
        x = torch.matmul(d, self.W_a)
        x = (x*e).sum(dim=2)
        x = torch.softmax(x, dim=0)
        return x

    def position(self, d, S):
        #output = [S, B]
        batch_size = d.shape[1]
        position_vect = torch.arange(S).unsqueeze(1).repeat(1, batch_size).float()
        if self.cuda:
            position_vect = position_vect.to('cuda')

        x = torch.matmul(d, self.W_p)
        x = torch.tanh(x)
        x = torch.matmul(x, self.v_p)
        x = torch.sigmoid(x)
        x = (S*x).squeeze()

        mask = (torch.abs(x - position_vect) <= self.D).float()
        #x = [1, B]
        #position_vect = [S, B]
        y = (x - position_vect)**2
        y = y/(2*(self.sigma**2))
        y = torch.exp(-y)*mask
        return y

    def forward(self, e, d):
        #e = [S, B, H]
        #d = [1, B, H]

        S = e.shape[0]
        batch_size = e.shape[1]

        a = self.alignment(e, d)
        p = self.position(d, S)

        alignment_weights = (a*p).reshape(S, batch_size, 1)
        context = (e*alignment_weights).sum(dim=0).unsqueeze(0)
        #context = [1, B, H]

        x = torch.cat([context, d], dim=2)
        output = torch.tanh(torch.matmul(x, self.W_c))
        output.shape
        alignment_weights.shape
        return output, alignment_weights.squeeze()
