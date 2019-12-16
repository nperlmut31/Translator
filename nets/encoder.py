import torch
import torch.nn as nn



class Encoder(nn.Module):

    def __init__(self, num_embeddings=5000, embedding_dim=100, hidden_size=100,
                 padding_idx = 2, dropout = 0.4):
        super().__init__()


        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim,
                                      padding_idx=padding_idx)

        self.GRU = nn.GRU(input_size=embedding_dim,
                          hidden_size=hidden_size,
                          num_layers=1,
                          bidirectional=True,
                          batch_first=False)

        self.dropout = nn.Dropout(p=dropout)


    @staticmethod
    def transform_hidden_state(h):
        b, s = h.shape[1], h.shape[0] * h.shape[2]
        h = h.permute(1, 0, 2)
        h = h.reshape(b, s)
        h = h.reshape(1, b, s)
        return h


    def forward(self, x):
        #input = [B, S, H]
        #output = [S, B, 2*H]
        #h-output = [1, B, 2*H]
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        output, h = self.GRU(x)
        h = self.transform_hidden_state(h)
        output = self.dropout(h)
        h = self.dropout(h)
        return output, h