
class SentimentAnalysisPar():
    vocab_size = 1000
    embedding_dim = 16
    max_length = 100
    num_epochs = 10
    split = 0.2
    max_subwords = 5
    focus = 'before'
    learning_rate = 0.0001

class TextGeneratorPar():
    embedding_dim = 64
    vocab_size = None # manually computed
    max_length = None # manually computed
    focus = 'after'
    cell_units = 20  
    num_epochs = 200
    learning_rate = 0.0001    
