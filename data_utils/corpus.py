import torch
from torch.utils.data import Dataset, DataLoader
import nltk
import nltk.tokenize as tokenizer
import json
import pickle
import os


dir_path = os.path.dirname(os.path.realpath(__file__))


def retrieve_sentences():

    p = os.path.join(dir_path, '..', 'data', 'train.json')
    with open(p, 'r') as f:
        data = json.load(f)
        f.close()

    source = [s[0] for s in data]
    target = [s[1] for s in data]
    return source, target


class Corpus(object):

    def __init__(self, sentences, min_count=2):
        self.count = self.get_count(sentences)
        self.word2idx = {'<sos>': 0, '<eos>': 1, '<pad>': 2, '<unk>': 3}
        self.idx2word = {0: '<sos>', 1: '<eos>', 2: '<pad>', 3: '<unk>'}
        counter = 4
        for w in self.count.keys():
            if self.count[w] >= min_count:
                self.word2idx[w] = counter
                self.idx2word[counter] = w
                counter += 1

    def get_count(self, sentences):
        count = {}
        for s in sentences:
            for w in s:
                if w in count:
                    count[w] += 1
                else:
                    count[w] = 1
        return count

    def make_dict(self):
        return {'count': self.count, 'word2idx': self.word2idx}

    def save_json(self, file_path):
        corpus = self.make_dict()
        with open(file_path, 'w') as f:
            json.dump(corpus,f)
            f.close()

    def save_pickle(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self,f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()

