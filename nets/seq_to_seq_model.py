import torch
import torch.nn as nn
from nets.encoder import Encoder
from nets.decoder import Decoder
from nets.attention_mechanism import Attention
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.functional import cross_entropy
import torch.nn.functional as f
import numpy as np



class OutputLayer(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        self.linear_layer = nn.Linear(in_features=input_size, out_features=output_size)

    def forward(self, x):
        x = x.squeeze()
        x = self.linear_layer(x)
        return x


class Seq2Seq_Att(nn.Module):

    def __init__(self, num_embeddings_source=1000, num_embeddings_target=1000,
                 embedding_dim=200, encoder_state_size=200, D=4, padding_idx=2,
                 cuda=False, use_attention=True, use_feedback=True, dropout=0.4):
        super().__init__()

        self.cuda = cuda
        self.use_attention = use_attention
        self.use_feedback = use_feedback
        if not self.use_attention:
            self.use_feedback = False


        self.encoder = Encoder(num_embeddings=num_embeddings_source,
                               embedding_dim=embedding_dim,
                               hidden_size=encoder_state_size,
                               padding_idx=padding_idx, dropout=dropout)

        self.decoder = Decoder(num_embeddings=num_embeddings_target,
                               embedding_dim=embedding_dim,
                               hidden_size=2*encoder_state_size,
                               padding_idx=padding_idx, cuda=self.cuda, dropout=dropout)


        self.output_layer = OutputLayer(input_size=2 * encoder_state_size, output_size=num_embeddings_target)


        if self.cuda:
            self.encoder = self.encoder.to('cuda')
            self.decoder = self.decoder.to('cuda')
            self.output_layer = self.output_layer.to('cuda')

        if self.use_attention:
            self.attention_mechanism = Attention(hidden_size=2*encoder_state_size,
                                                 D=D, cuda=self.cuda, dropout=dropout)
            if self.cuda:
                self.attention_mechanism.to('cuda')

