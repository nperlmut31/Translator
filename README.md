# Translator

The repo contains a neural machine translator. The architecture contains four main components: encoder RNN, decoder RNN, and attention mechanism. The output sequences are decoded using beam search. The model is trained using "teacher forcing".

The model's design is inspired by the contents of the paper: [Affective Approaches to Attention-based Neural Machine Translation](https://www-nlp.stanford.edu/pubs/emnlp15_attn.pdf). 

This repo was one of my first deep learning projects and I built it in order to learn about sequence-to-sequence models and NLP.

To train the model run the file train.py. The specifications and hyperparameters are contained in params.json. 

