# -*- coding: utf-8 -*-

# Import Tokenizer and pad_sequences
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Import numpy and pandas
import numpy as np
import pandas as pd

def get_input(data, test_split=0.2):
    sentences = data['text'].tolist()
    labels = data['sentiment'].tolist()

    # Separate out the sentences and labels into training and test sets
    training_size = int(len(sentences) * (1-test_split))

    train_sentences = sentences[0:training_size]
    test_sentences = sentences[training_size:]
    train_labels = np.array(labels[0:training_size])
    test_labels = np.array(labels[training_size:])

    return (train_sentences, train_labels), (test_sentences, test_labels) 

def get_data(fname, source, raw=True):
    path = tf.keras.utils.get_file(fname, url)
    dataset = pd.read_csv(path)
    # print(dataset.head())
    if raw:
        return dataset
    else:
        return get_input(dataset)

def to_sequence(vocab_source, 
                target='same', 
                vocab_size=None, 
                maxlen=None, 
                focus='after'):
    manip='pre' if focus=='after' else 'post'            
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(vocab_source)
    word_index = tokenizer.word_index
    target = vocab_source if target=='same' else target
    seq = tokenizer.texts_to_sequences(target)
    seq = pad_sequences(seq, maxlen=None, padding=manip, truncating=manip)
    return seq, word_index

def decode(text, reverse_dict):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
 

