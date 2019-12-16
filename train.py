from data_utils.dataset import TextDataset
from data_utils.corpus import Corpus, retrieve_sentences
from nets.seq_to_seq_model import Seq2Seq_Att
from nets.teacher_forcing import TeacherForcing
from nets.beam_search import Translate

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from nltk.translate import bleu_score
from utils.utils import scoring_preprocess
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


p = os.path.join(dir_path, 'params.json')
with open(p, 'r') as f:
    params = json.load(f)
    f.close()


embedding_dim = params['embedding_dim']
encoder_state_size = params['encoder_state_size']
D = params['D']
dropout = params['dropout']
lr = params['lr']
batch_size = params['batch_size']
epochs = params['epochs']
beam_size = params['beam_size']
cuda = params['cuda']
load_corpus = params['load_corpus']
num_workers = params['num_workers']
run_tests = params['run_tests']
trainingBLEU = params['trainingBLEU']
use_attention = params['use_attention']
use_feedback = params['use_feedback']



if params['train_index_range'] == [0, 0]:
    train_index_range = None
else:
    train_index_range = params['train_index_range']

if params['test_index_range'] == [0, 0]:
    test_index_range = None
else:
    test_index_range = params['test_index_range']


target_sentences, source_sentences = retrieve_sentences()
source_corpus = Corpus(sentences=source_sentences)
target_corpus = Corpus(sentences=target_sentences)

p = os.path.join(dir_path, 'data', 'german_corpus.pkl')
q = os.path.join(dir_path, 'data', 'english_corpus.pkl')
source_corpus.save_pickle(file_path=p)
target_corpus.save_pickle(file_path=q)


dataset = TextDataset(mode='train',
                      source_corpus=source_corpus,
                      target_corpus=target_corpus,
                      index_range=train_index_range)

loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn, num_workers=num_workers)


source_count = dataset.source_word_count()
target_count = dataset.target_word_count()


model = Seq2Seq_Att(num_embeddings_source=source_count, num_embeddings_target=target_count,
                 embedding_dim=embedding_dim, encoder_state_size=encoder_state_size,
                    D=D, use_attention=use_attention, use_feedback=use_feedback,
                    cuda=cuda, dropout=dropout)


trainer = TeacherForcing(model=model, cuda=cuda)
optimizer = optim.Adam(trainer.parameters(), lr=lr)


record_BLEU = 0
for epoch in range(epochs):
    for i, batch in enumerate(loader):

        source, target = batch
        if cuda:
            source = (source[0].to('cuda'), source[1].to('cuda'))
            target = (target[0].to('cuda'), target[1].to('cuda'))


        if trainingBLEU and i%50 == 0:
            translator = Translate(model=trainer.model, beam_size=beam_size, cuda=cuda)
            output = translator(source)
            output_trans = scoring_preprocess(output, corpus=target_corpus, input_type='hypothesis')
            ground_truth = scoring_preprocess(target[0], corpus=target_corpus, input_type='target')
            BLEU4 = bleu_score.corpus_bleu(ground_truth, output_trans, weights=(0.0, 0.0, 0.0, 1.0))
            print('batch BLEU score: ' + str(BLEU4), flush=True)


        optimizer.zero_grad()
        loss = trainer(source, target)
        loss.backward()
        optimizer.step()
        if i%10 == 0:
            print((epoch, i, loss), flush=True)


    if run_tests:
        test_dataset = TextDataset(mode='test', index_range=test_index_range)
        test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             collate_fn=test_dataset.collate_fn,
                             num_workers=num_workers)
        translator = Translate(model=trainer.model, beam_size=beam_size, cuda=cuda)

        total_predictions = []
        total_targets = []
        for batch in test_loader:
            source, target = batch
            if cuda:
                source = (source[0].to('cuda'), source[1].to('cuda'))
                target = (target[0].to('cuda'), target[1].to('cuda'))
            output = translator(source)

            output = scoring_preprocess(output, corpus=target_corpus, input_type='hypothesis')
            ground_truth = scoring_preprocess(target[0], corpus=target_corpus, input_type='target')

            total_predictions += output
            total_targets += ground_truth

        BLEU4 = bleu_score.corpus_bleu(total_targets, total_predictions, weights=(0.0, 0.0, 0.0, 1.0))
        print('Test BLEU score: ' + str(BLEU4), flush=True)

        if BLEU4 > record_BLEU:
            record_BLEU = BLEU4
            print('New record!', flush=True)

            p = os.path.join(dir_path, 'model_state_dicts', 'model_state_dict.pt')
            with open(p, 'wb') as f:
                torch.save(model.state_dict(), f)
                f.close()
        else:
            print('This result is less than current record.', flush=True)
