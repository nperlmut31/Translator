import torch
import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self, num_embeddings=5000, embedding_dim=100, hidden_size=200,
                 padding_idx=2, use_feedback=True, cuda = False, dropout = 0.4):
        super().__init__()

        self.use_feedback = use_feedback
        self.cuda = cuda

        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=padding_idx)

        if self.use_feedback:
            input_size = embedding_dim + hidden_size
        else:
            input_size = embedding_dim

        self.GRU = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=1, bidirectional=False, batch_first=False)

        self.dropout = nn.Dropout(p=dropout)


    def forward(self, *args):
        x, h = args[0], args[1]
        x = self.embedding(x)
        x = x.permute(1, 0, 2)

        if self.use_feedback:
            if len(args) == 2:
                feedback = torch.zeros(*h.shape).float()
                if self.cuda:
                    feedback = feedback.to('cuda')

            else:
                feedback = args[2]
            x = torch.cat([x, feedback], dim=2)
            output, h = self.GRU(x, h)
        else:
            output, h = self.GRU(x, h)

        output = self.dropout(output)
        h = self.dropout(h)
        return output, h