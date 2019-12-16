import torch
from torch.utils.data import Dataset, DataLoader
import nltk
import nltk.tokenize as tokenizer
import json
import pickle
import os


dir_path = os.path.dirname(os.path.realpath(__file__))


class TextDataset(Dataset):

    def __init__(self, source_corpus, target_corpus, mode='train', index_range = None):
        super().__init__()

        self.index_range = index_range

        self.source_corpus = source_corpus
        self.target_corpus = target_corpus

        if mode == 'train':
            path = os.path.join(dir_path, '..', 'data', 'train.json')
        else:
            path = os.path.join(dir_path, '..', 'data', 'test.json')

        target_sentences, source_sentences = self.load_sentences(path)
        sorted_lengths = sorted([(len(source_sentences[i]), i) for i in range(len(source_sentences))])
        source_sentences = [source_sentences[x[1]] for x in sorted_lengths]
        target_sentences = [target_sentences[x[1]] for x in sorted_lengths]
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences

    def __getitem__(self, index):
        source_sentence = list(map(lambda w: self.source_corpus.word2idx[w], self.source_sentences[index]))
        target_sentence = list(map(lambda w: self.target_corpus.word2idx[w], self.target_sentences[index]))
        return source_sentence, target_sentence

    def collate_fn(self, batch):
        longest_length_source = max([len(x[0]) for x in batch])
        longest_length_target = max([len(x[1]) for x in batch])
        source_lengths = []
        target_lengths = []
        padded_source = []
        padded_target = []

        pad_label_s = self.source_corpus.word2idx['<pad>']
        pad_label_t = self.target_corpus.word2idx['<pad>']
        for b in batch:
            p_s = b[0] + [pad_label_s for i in range(longest_length_source - len(b[0]))]
            p_t = b[1] + [pad_label_t for i in range(longest_length_target - len(b[1]))]
            padded_source.append(p_s)
            padded_target.append(p_t)
            source_lengths.append(len(b[0]))
            target_lengths.append(len(b[1]))

        padded_source = torch.tensor(padded_source).long()
        source_lengths = torch.tensor(source_lengths).long()
        padded_target = torch.tensor(padded_target).long()
        target_lengths = torch.tensor(target_lengths).long()

        return (padded_source, source_lengths), (padded_target, target_lengths)

    def __len__(self):
        return len(self.source_sentences)

    def source_word_count(self):
        return len(self.source_corpus.word2idx.keys())

    def target_word_count(self):
        return len(self.target_corpus.word2idx.keys())

    def load_sentences(self, path):

        def replace_unk_source(w):
            if w not in self.source_corpus.word2idx:
                return '<unk>'
            else:
                return w

        def replace_unk_target(w):
            if w not in self.target_corpus.word2idx:
                return '<unk>'
            else:
                return w

        data = json.load(open(path, 'r'))
        if self.index_range:
            data = data[self.index_range[0]: self.index_range[1]]
        source_sentences = []
        target_sentences = []
        for x in data:
            t, s = list(map(replace_unk_target, x[0])), list(map(replace_unk_source, x[1]))
            target_sentences.append(['<sos>'] + t + ['<eos>'])
            source_sentences.append(['<sos>'] + s + ['<eos>'])
        return target_sentences, source_sentences


    def create_test_batch(self, index_range=(0, 5)):
        b = [self.__getitem__(i) for i in range(index_range[0], index_range[1])]
        return self.collate_fn(b)


    def save_test_batch(self, file_path, index_range=(0, 5)):
        batch = self.create_test_batch(index_range=index_range)
        with open(file_path, 'wb') as f:
            torch.save(batch, f)
            f.close()