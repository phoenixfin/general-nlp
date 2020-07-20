import tensorflow as tf

import numpy as np
import pandas as pd

import preprocess as pp
import support
from basic import BasicNLP
import parameters as p

dataset = pp.get_data('reviews.csv', p.url)
sentences, labels = pp.split_data(dataset, test_split=p.split)
sequences, tokens = pp.get_input_sequences(*sentences, 
                                            vocab_size=p.vocab_size, 
                                            maxlen=p.max_length, 
                                            focus=p.focus)

print(pp.decode(sequences[0][1], tokens))

NLP = BasicNLP('binary')
print(len(sequences[0]))
NLP.set_sequence_data(*sequences)
NLP.set_label_data(*labels)
NLP.add_embedding(p.vocab_size, p.embedding_dim, p.max_length)
NLP.add_dense([6])
NLP.add_output()
NLP.get_summary()
NLP.train(p.num_epochs, optimizer='adam')

"""## Predicting Sentiment in New Reviews

Now that you've trained and visualized your network, take a look below at how we can predict sentiment in new reviews the network has never seen before.
"""

# Use the model to predict a review   
fake_reviews = ['I love this phone', 'I hate spaghetti', 
                'Everything was cold',
                'Everything was hot exactly as I wanted', 
                'Everything was green', 
                'the host seated us immediately',
                'they gave us free chocolate cake', 
                'not sure about the wilted flowers on the table',
                'only works when I stand on tippy toes', 
                'does not work when I stand on my head']

NLP.predict(fake_reviews, tokens)