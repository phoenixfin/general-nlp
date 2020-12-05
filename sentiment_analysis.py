from basic import BasicNLP
import preprocess as pp
import support
from parameters import SentimentAnalysisPar as p
import libraries as lib

class SentimentAnalysis(BasicNLP):
    mode = 'binary'
    data_name = 'reviews'
    
    def __init__(self, use_subwords=False):
        super().__init__()
        self.use_subwords = use_subwords

    def _basic_embedding(self):
        self.add_squasher('flatten')
        self.add_dense([6])    

    def _convolutional(self):
        self.add_convolution(16, 5)

    def _simple_rnn(self):
        self.add_recurrent_cell('SimpleRNN')
        
    def _single_LSTM(self):
        self.add_recurrent_cell('LSTM', bidirectional=True)
    
    def _multi_LSTM(self):
        self.add_recurrent_cell('LSTM', bidirectional=True, double=True)
        
    def _single_GRU(self):
        self.add_recurrent_cell('GRU', bidirectional=True)        

    def _predicate(self, score):
        if score < 0.2:
            return 'worst!'
        if score < 0.4:
            return 'quite bad'
        if score < 0.6:
            return 'hard to decide'
        if score < 0.8:
            return 'quite good'
        return 'excellent'

    def output(self):
        test = lib.test_reviews[0]
        predicted = self.predict(test, tokens)
        for i in range(len(test_data)):
            pred = self._predicate(predicted[i])
            print('{} : {} ({})'.format(test[i], pred, predicted[i]))

    def setup(self):
        sentences, labels = pp.split_data(self.raw_data, test_split=p.split)
        sub = 5 if self.use_subwords else 0    
        sequences, tokens = pp.get_input_sequences(*sentences, 
                                                    vocab_size=p.vocab_size, 
                                                    maxlen=p.max_length, 
                                                    focus=p.focus,
                                                    subwords=sub)
        self.tokens = tokens
        return sequences, labels

    

    def main(self, method):
        sequences, labels = self.setup()
        self.set_sequence_data(*sequences)
        self.set_label_data(*labels)
        self.add_embedding(p.vocab_size, p.embedding_dim, p.max_length)

        getattr(self, '_'+method)()

        self.add_output()
        self.get_summary()
        self.train(p.num_epochs, 'adam')
        