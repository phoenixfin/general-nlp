import tensorflow.keras as tk

class BasicNLP(object):
    def __init__(self, mode):
        self.model = tk.Sequential()
        self.mode = 'binary'
        temp = {'sequence':None, 'label':None}
        self.data = {'train':temp, 'test':temp}

    def add_embedding(self, vocab_size, dimension, input_length):
        layer = tk.layers.Embedding(vocab_size, dimension, 
                                    input_length=input_length)
        self.model.add(layer)
        self.model.add(tk.layers.Flatten())
        
    def add_squasher(self, method):
        if method == 'flatten':
            layer_typ = 'Flatten'
        elif method == 'average_pool':
            layer_typ = 'GlobalAveragePooling1D'
        elif method == 'max_pool':
            layer_typ = 'GlobalMaxPooling1D'
        layer = getattr(tk.layers, layer_typ)
        self.model.add(layer())   
    
    def add_dense(self, neurons_list, activation='relu'):
        for neurons in neurons_list:
            self.model.add(tk.layers.Dense(neurons, activation=activation))

    def add_output(self):
        if self.mode=='binary':
            activation = 'sigmoid'
            neurons = 1
        else:
            activation = 'softmax'
            neurons = None
        self.add_dense([neurons], activation)
    
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
        self.model.fit(train['sequence'], train['label'],
                       epochs=num_epochs, 
                       validation_data=(test['sequence'], test['label']))
        
    def predict(self, texts, tokenizer):
        import preprocess, parameters
        sequences, _ = preprocess.to_sequence(
            target = texts,
            vocab_size=parameters.vocab_size, 
            maxlen=parameters.max_length, 
            focus=parameters.focus
        )
        classes = ['positive', 'negative']
        predicted = self.model.predict(sequences)
        
        # The closer the class is to 1, the more positive the review is deemed to be
        for x in range(len(texts)):
            print(texts[x],':',predicted[x])
            print('\n')
