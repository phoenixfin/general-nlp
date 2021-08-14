import tensorflow.keras as tk
from support import plot_graphs
import libraries as lib
import preprocess as pp

class BasicNLP(object):
    def __init__(self, data_num=None):
        self.model = tk.Sequential()
        self.data = {'train':{}, 'test':{}}
        self.tokens = None
        self.history = None
        fname = self.data_name + '.csv'
        self.raw_data = pp.get_data(fname, lib.data[self.data_name], data_num)        

    def add_embedding(self, vocab_size, dimension, input_length):
        self.embedding_dimension = dimension
        layer = tk.layers.Embedding(vocab_size, dimension, 
                                    input_length=input_length)
        self.model.add(layer)
        
    def add_squasher(self, method):
        if method == 'flatten':
            layer_typ = 'Flatten'
        elif method == 'average_pool':
            layer_typ = 'GlobalAveragePooling1D'
        elif method == 'max_pool':
            layer_typ = 'GlobalMaxPooling1D'
        layer = getattr(tk.layers, layer_typ)
        self.model.add(layer())   
    
    def add_convolution(self, filters, kernel_size, activation='relu'):
        layer = tk.layers.Conv1D(filters, kernel_size, activation=activation) 
        self.model.add(layer)       
    
    def add_dense(self, neurons_list, activation='relu'):
        for neurons in neurons_list:
            self.model.add(tk.layers.Dense(neurons, activation=activation))

    def add_output(self):
        if self.mode=='binary':
            activation = 'sigmoid'
            neurons = 1
        else:
            activation = 'softmax'
            neurons = len(self.tokens.word_index) + 1
        self.add_dense([neurons], activation)
    
    def add_recurrent_cell(self, typ, units=None, bidirectional=False, double=False):
        def add(cell):
            layer = tk.layers.Bidirectional(cell) if bidirectional else cell
            self.model.add(layer)

        if units==None:
            units = self.embedding_dimension            
            if typ == 'GRU': units *= 2
        cl = getattr(tk.layers, typ)
        if double:
            add(cl(units, return_sequences=True))
        add(cl(units))

        
    def add_dropout(self, rate):
        self.model.add(tk.layers.Dropout(rate))
    
    def get_summary(self):
        self.model.summary()
        
    def set_train_data(self, train_seq, train_label):
        self.data['train']['sequence'] = train_seq
        self.data['train']['label'] = train_label

    def set_test_data(self, test_seq, test_label):
        self.data['test']['sequence'] = test_seq
        self.data['test']['label'] = test_label
        
    def set_sequence_data(self, train_seq, test_seq):
        self.data['train']['sequence'] = train_seq
        self.data['test']['sequence'] = test_seq        
        
    def set_label_data(self, train_label, test_label):
        self.data['train']['label'] = train_label
        self.data['test']['label'] = test_label        
        
    def train(self, num_epochs, optimizer):
        loss = self.mode + '_crossentropy'
        self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])        
        train = self.data['train']
        test = self.data['test']
        val_data = (test['sequence'], test['label']) if test else None 
        history = self.model.fit(train['sequence'], train['label'],
                                epochs=num_epochs, 
                                validation_data=val_data)
        self.history = history
        
    def predict(self, texts):
        import preprocess, parameters        
        p = getattr(parameters, self.__class__.__name__ +'Par')
        sequences, _ = preprocess.to_sequence(
            target = texts,
            tokenizer = self.tokens,
            vocab_size=p.vocab_size, 
            maxlen=p.max_length, 
            focus=p.focus
        )
        predicted = self.model.predict(sequences)
        return predicted

    def plot(self, metric):
        if self.history:
            plot_graphs(self.history, metric) 
        else:
            pass