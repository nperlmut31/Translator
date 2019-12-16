import torch
import torch.nn as nn
import torch.nn.functional as f


class TeacherForcing(nn.Module):

    def __init__(self, model, cuda=False, PAD=2, EOS=1):
        super().__init__()

        self.cuda = cuda
        self.model = model
        self.EOS = EOS
        self.PAD = PAD


    def loss(self, input, target, mask):
        target = target.squeeze()

        input = input[mask,:]
        target = target[mask]

        cross_entropy = f.cross_entropy(input, target)
        #l = mask.float().sum()
        return cross_entropy


    def forward(self, input, output):
        input, input_lengths = input[0], input[1]
        target, target_lengths = output[0], output[1]

        batch_size, sequence_length = target.shape[0], target.shape[1]

        encoder_output, hidden_state = self.model.encoder(input)
        total_loss = torch.tensor(0.0).float()

        if self.cuda:
            total_loss = total_loss.to('cuda')

        for i in range(sequence_length-1):
            mask = ((target[:,i] != self.EOS) & (target[:,i] != self.PAD))

            if i > 0 and self.model.use_feedback:
                decoder_args = [target[:, i:i+1], hidden_state, feedback]
            else:
                decoder_args = [target[:,i:i+1], hidden_state]

            output, hidden_state = self.model.decoder(*decoder_args)
            if self.model.use_attention:
                output = self.model.attention_mechanism(encoder_output, output)[0]
                feedback = output

            output = self.model.output_layer(output)
            total_loss += self.loss(output, target[:,i+1], mask)

        return total_loss
