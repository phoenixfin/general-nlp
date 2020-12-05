import tensorflow as tf
import numpy as np
import pandas as pd

from sentiment_analysis import SentimentAnalysis
from text_generator import TextGenerator
    
if __name__ == '__main__':
    obj = SentimentAnalysis()
    obj.main('multi_LSTM')

    
    # obj = TextGenerator()
    # obj.main()
    # obj.output("im feeling chills", 100)