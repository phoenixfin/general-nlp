# -*- coding: utf-8 -*-

from string import punctuation as punc
import tensorflow as tf
import numpy as np

from parameters import TextGeneratorPar as p
from basic import BasicNLP
import preprocess as pp

class TextGenerator(BasicNLP):
    mode = 'categorical'
    data_name = 'song'
    
    def __init__(self):
        super().__init__(data_num = 10)

    def _create_corpus(self):
        txt = self.raw_data['text']
        
        txt = txt.str.replace('[{}]'.format(punc), '')
        txt = txt.str.lower()
        corpus = txt.str.cat().split('\n')

        for l in range(len(corpus)):
            corpus[l] = corpus[l].rstrip()

        corpus = [l for l in corpus if l != '']
        return corpus

    def setup(self):
        corpus = self._create_corpus()
        self.tokens = pp.tokenize(corpus)

        sequences = []
        for line in corpus:
            token_list = self.tokens.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                sequences.append(n_gram_sequence)

        padder = tf.keras.preprocessing.sequence.pad_sequences
        sequences = np.array(padder(sequences, padding='pre'))
        input_sequences, labels = sequences[:,:-1], sequences[:,-1]
        total_words = len(self.tokens.word_index) + 1
        one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=total_words)

        return input_sequences, one_hot_labels        
        
    def main(self):
        sequences, labels = self.setup()
        max_len = max([len(seq) for seq in sequences])-1
        self.set_train_data(sequences, labels)
        total_words = len(self.tokens.word_index) + 1
        self.add_embedding(total_words, p.embedding_dim, max_len)
        self.add_recurrent_cell('LSTM', units=p.cell_units, bidirectional=True)
        self.add_output()
        
        self.max_length = max_len
        # self.train(p.num_epochs, optimizer='adam')

    def output(self, seed_text, next_words):        
        for _ in range(next_words):
            scores = self.predict(seed_text)
            predicted = np.argmax(scores, axis=-1)
            output_word = pp.decode(predicted, self.tokens)
            seed_text += " " + output_word

        print(seed_text)