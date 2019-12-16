import torch
import torch.nn as nn
from torch.nn.functional import log_softmax


class Beam(object):

    def __init__(self, beam_size, cuda=False, SOS=0, EOS=1, PAD=2):

        self.PAD = PAD
        self.SOS = SOS
        self.EOS = EOS
        self.cuda = cuda

        if self.cuda:
            self.active_sequences = torch.zeros(beam_size, 1).long().to('cuda')
            self.active_likelihoods = torch.zeros(beam_size).float().to('cuda')
            self.next_inputs = torch.zeros(beam_size, 1).to('cuda')
            self.completed_sequences = torch.tensor([]).to('cuda')
            self.completed_likelihoods = torch.tensor([]).to('cuda')
            self.decomp_params = [beam_size, torch.arange(beam_size).to('cuda')]
        else:
            self.active_sequences = torch.zeros(beam_size, 1).long()
            self.active_likelihoods = torch.zeros(beam_size).float()
            self.next_inputs = torch.zeros(beam_size, 1)
            self.completed_sequences = torch.tensor([])
            self.completed_likelihoods = torch.tensor([])
            self.decomp_params = [beam_size, torch.arange(beam_size)]

        pass

    def initial_advance(self, word_probs):
        search_size = self.active_sequences.shape[0]
        likelyhooods = word_probs[0]

        L_top = torch.argsort(likelyhooods, dim=0, descending=True)[:search_size]
        top_likelyhoods = likelyhooods[L_top]

        self.active_sequences = torch.cat([self.active_sequences, L_top.unsqueeze(1)], dim=1)
        self.active_likelihoods = self.active_likelihoods + top_likelyhoods
        self.next_inputs = self.active_sequences[:, -1].unsqueeze(1)


    def advance(self, word_probs):
        search_size = len(self.active_sequences)
        vocab_count = word_probs.shape[1]

        #get top k elements of log_probs
        L = (self.active_likelihoods.unsqueeze(1) + word_probs).reshape(-1,)
        L_top = torch.argsort(L, dim=0, descending=True)[:search_size]

        seq_ind = (L_top // vocab_count)
        word_ind = (L_top % vocab_count).unsqueeze(1)

        candidate_sequences = torch.cat([self.active_sequences[seq_ind], word_ind], dim=1)
        candidate_likelihoods = L[L_top]

        active_mask = (candidate_sequences[:, -1] != self.EOS)
        completed_mask = (candidate_sequences[:, -1] == self.EOS)

        new_completed_sequences = candidate_sequences[completed_mask, :]
        length = new_completed_sequences.shape[1]
        new_completed_likelihoods = torch.masked_select(candidate_likelihoods, completed_mask)/length

        if self.completed_sequences.shape[0] == 0:
            if completed_mask.sum() > 0:
                self.completed_sequences = new_completed_sequences
                self.completed_likelihoods = new_completed_likelihoods
        else:
            l = self.completed_sequences.shape[0]
            if self.cuda:
                padding = torch.tensor([self.PAD]).long().repeat(l, 1).to('cuda')
            else:
                padding = torch.tensor([self.PAD]).long().repeat(l, 1)
            self.completed_sequences = torch.cat([self.completed_sequences, padding], dim=1)

            a, b = self.completed_sequences, self.completed_likelihoods
            self.completed_sequences = torch.cat([a, new_completed_sequences], dim=0)
            self.completed_likelihoods = torch.cat([b, new_completed_likelihoods], dim=0)

        self.active_sequences = candidate_sequences[active_mask, :]
        self.active_likelihoods = torch.masked_select(candidate_likelihoods, active_mask)
        self.decomp_params = [self.decomp_params[1].shape[0], torch.masked_select(seq_ind, active_mask)]
        self.next_inputs = self.active_sequences[:, -1].unsqueeze(1)


    def return_winner(self):
        if self.completed_sequences.shape[0] > 0 and self.completed_likelihoods.shape[0] > 0:
            ind = self.completed_likelihoods.argmax()
            return self.completed_sequences[ind]
        else:
            ind = self.active_likelihoods.argmax()
            return self.active_sequences[ind]

    def num_active_sequences(self):
        return len(self.active_sequences)


class Translate(nn.Module):

    def __init__(self, model, beam_size, cuda=False):
        super().__init__()

        self.beam_size = beam_size
        self.model = model
        self.beams = None
        self.SOS = 0
        self.EOS = 1
        self.PAD = 2
        self.cuda = cuda

        if self.cuda:
            self.model.to('cuda')

    def forward(self, source):
        if isinstance(source, tuple):
            source = source[0]

        encoder_state, hidden_state = self.model.encoder(source)
        batch_size = hidden_state.shape[1]
        #must now appropriately reshape hiddenstate h
        hidden_state = self.initialize_h(hidden_state)
        if self.model.use_attention:
            encoder_state = self.initialize_h(encoder_state)

        #initialize the beams
        self.beams = [Beam(beam_size=self.beam_size, cuda=self.cuda) for i in range(batch_size)]
        counter = 0
        while (sum([beam.num_active_sequences() for beam in self.beams]) > 0) \
                and (counter < source.shape[1] + 1):

            input = self.arrange_input()
            hidden_state = self.arrange_h(hidden_state=hidden_state)

            if self.model.use_feedback and counter > 0:
                feedback = self.arrange_h(hidden_state=feedback)
                decoder_args = [input, hidden_state, feedback]
            else:
                decoder_args = [input, hidden_state]

            x, hidden_state = self.model.decoder(*decoder_args)
            if self.model.use_attention:
                encoder_state = self.arrange_h(hidden_state=encoder_state)
                x, alignment_weights = self.model.attention_mechanism(encoder_state, x)
                feedback = x

            output = self.model.output_layer(x)

            if len(output.shape) < 2:
                output = output.unsqueeze(0)
            output = log_softmax(input=output, dim=1, dtype=torch.float)
            output_list = self.decompose_output(output)

            for i, beam in enumerate(self.beams):
                if counter == 0:
                    beam.initial_advance(output_list[i])
                else:
                    beam.advance(output_list[i])
            counter += 1

        translations = []
        for beam in self.beams:
            translations.append(beam.return_winner())

        return translations


    def arrange_h(self, hidden_state):
        h = hidden_state
        hidden_states = []
        for i, beam in enumerate(self.beams):
            size, indices = beam.decomp_params[0], beam.decomp_params[1]
            h_1, h_2 = h[:,:size,:][:,indices,:], h[:,size:,:]
            hidden_states.append(h_1)
            h = h_2
        hidden_states = torch.cat(hidden_states, dim=1)
        return hidden_states


    def arrange_input(self):
        input_list = []
        for beam in self.beams:
            input_list.append(beam.next_inputs)
        input_list = torch.cat(input_list, dim=0).long()
        return input_list


    def decompose_output(self, output):
        output_list = []
        for beam in self.beams:
            size = len(beam.active_sequences)
            output_1, output_2 = output[:size], output[size:]
            output_list.append(output_1)
            output = output_2
        return output_list


    def initialize_h(self, h):
        #b, s = h.shape[1], h.shape[0] * h.shape[2]
        #h = h.permute(1, 0, 2)
        #h = h.reshape(1, b, s)
        b = h.shape[1]
        h_slices = []
        for i in range(b):
            h_slices.append(h[:,i:i+1,:].repeat(1, self.beam_size, 1))
        h = torch.cat(h_slices, dim=1)
        return h

