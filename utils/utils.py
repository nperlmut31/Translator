import torch


def add_padding(self, predictions):
    max_length = max(map(lambda x: len(x), predictions))
    padded_predictions = []
    for p in predictions:
        if max_length - p.shape[0] > 0:
            pad = torch.tensor([self.PAD]).repeat(max_length - p.shape[0])
            padded_p = torch.cat([p, pad], dim=0).unsqueeze(0)
            padded_predictions.append(padded_p)
        else:
            padded_predictions.append(p.unsqueeze(0))
    padded_predictions = torch.cat(padded_predictions, dim = 0)
    if self.cuda:
        padded_predictions.to('cuda')
    return padded_predictions


def add_buffer(target, output, cuda=False):
    batch_size = output.shape[0]
    ground_truth = target[0]
    length_diff = abs(output.shape[1] - ground_truth.shape[1])
    if cuda:
        buffer = torch.tensor([[2 for i in range(length_diff)] for i in range(batch_size)]).long().to('cuda')
    else:
        buffer = torch.tensor([[2 for i in range(length_diff)] for i in range(batch_size)]).long()
    if output.shape[1] > ground_truth.shape[1]:
        ground_truth = torch.cat([ground_truth, buffer], dim=1)
    elif output.shape[1] < ground_truth.shape[1]:
        output = torch.cat([output, buffer], dim=1)
    return ground_truth, output


def scoring_preprocess(sequences, corpus, input_type='target'):
    sequence_list = []
    for s in sequences:
        w = s[(s != 0) & (s != 1) & (s != 2)]
        L = list(map(lambda x: corpus.idx2word[int(x)], w))
        if input_type == 'target':
            sequence_list.append([L])
        else:
            sequence_list.append(L)
    return sequence_list
