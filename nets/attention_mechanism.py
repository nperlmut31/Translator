
import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, hidden_size = 100, D = 2, cuda=False, alt_ending = True, dropout=0.4):
        super().__init__()

        self.alt_ending = alt_ending
        self.cuda = cuda
        self.D = D
        self.sigma = self.D/2

        self.W_a = nn.Parameter(torch.rand(hidden_size, hidden_size, requires_grad=True))
        self.W_p = nn.Parameter(torch.rand(hidden_size, hidden_size, requires_grad=True))
        self.v_p = nn.Parameter(torch.rand(hidden_size, 1, requires_grad=True))
        self.W_c = nn.Parameter(torch.rand(2*hidden_size, hidden_size, requires_grad=True))
        self.linear_layer = nn.Linear(2*hidden_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(num_features=hidden_size)

        self.dropout = nn.Dropout(p=dropout)

        for p in self.named_parameters():
            if 'W' in p[0]:
                nn.init.orthogonal_(p[1])

    def alignment(self, e, d):
        #Mathematically this is:  softmax([< d*W_a, e_1>, ..., < d*W_a, e_s>])
        x = torch.matmul(d, self.W_a)
        x = x*e
        x = x.sum(dim=2)
        y = x.shape
        x = torch.softmax(x, dim=0)
        return x

    def position(self, d, S):
        batch_size = d.shape[1]
        position_vect = torch.arange(S).unsqueeze(1).repeat(1, batch_size).float()
        if self.cuda:
            position_vect = position_vect.to('cuda')

        x = torch.matmul(d.squeeze(), self.W_p)
        x = torch.tanh(x)
        x = torch.matmul(x, self.v_p).squeeze()
        x = x.reshape(1, -1)
        x = torch.sigmoid(x)
        x = S*x

        mask = (torch.abs(x - position_vect) <= self.D).float()
        y = (x - position_vect)**2
        y = y/(2*(self.sigma**2))
        y = torch.exp(-y)*mask
        return y


    def forward(self, e, d):
        S = e.shape[0]
        batch_size = e.shape[1]

        a = self.alignment(e, d)
        p = self.position(d, S)

        alignment_weights = (a*p).reshape(S, batch_size, 1)
        context = (e*alignment_weights).sum(dim=0)
        context = context.unsqueeze(0)

        x = torch.cat([context, d], dim=2)

        if self.alt_ending:
            x = self.linear_layer(x)
            x = nn.ReLU()(x)

            output = self.dropout(x)
        else:
            x = torch.tanh(torch.matmul(x, self.W_c))
            output = self.dropout(x)
        return output, alignment_weights.squeeze()