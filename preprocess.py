# -*- coding: utf-8 -*-
import tensorflow_datasets as tfds
import tensorflow.keras as tk
import numpy as np
import pandas as pd

def split_data(data, test_split=0.2):
    sentences = data['text'].tolist()
    labels = data['sentiment'].tolist()

    # Separate out the sentences and labels into training and test sets
    training_size = int(len(sentences) * (1-test_split))

    train_sentences = sentences[0:training_size]
    test_sentences = sentences[training_size:]
        
    train_labels = np.array(labels[0:training_size])
    test_labels = np.array(labels[training_size:])

    return (train_sentences, test_sentences), (train_labels, test_labels) 

def get_data(fname, source, num=None, raw=True):

    path = tk.utils.get_file(fname, source)
    dataset = pd.read_csv(path)
    if num:
        dataset=dataset[:num]
    # print(dataset.head())
    if raw:
        return dataset
    else:
        return get_input(dataset)

def tokenize(vocab_source, vocab_size=None, max_sub=0):
    if max_sub:
        SubEncoder = tfds.features.text.SubwordTextEncoder
        tokenizer = SubEncoder.build_from_corpus(vocab_source, 
                                                 vocab_size, 
                                                 max_subword_length=max_sub)
    else:
        tokenizer = tk.preprocessing.text.Tokenizer(num_words=vocab_size, 
                                                    oov_token="<OOV>")
        tokenizer.fit_on_texts(vocab_source)
    return tokenizer

def to_sequence(target, tokenizer=None, vocab_size=None, 
                maxlen=None, focus='after', subwords=0):
    manip='pre' if focus=='after' else 'post'        

    if tokenizer == None: 
        tokenizer = tokenize(target, vocab_size=vocab_size, max_sub=subwords)

    seq = tokenizer.texts_to_sequences(target)
    seq = tk.preprocessing.sequence.pad_sequences(seq, maxlen=maxlen, 
                                                  padding=manip, 
                                                  truncating=manip)
    return seq, tokenizer

def get_input_sequences(train_sen, test_sen, **kwargs):
    train_sequences, tokens = to_sequence(train_sen, **kwargs)
    test_sequences, _ = to_sequence(test_sen, tokenizer=tokens, **kwargs)
    return (train_sequences, test_sequences), tokens

def decode(text, tokenizer):
    dictionary = tokenizer.word_index
    reverse_dict = ['']+list(dictionary.keys())
    raw = [reverse_dict[i] for i in text]
    clean_text = ' '.join(raw).strip()
    return clean_text
 