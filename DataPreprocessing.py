import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
nltk.download('stopwords')

class DataPreprocessing:

    def __init__(self):
        print("DataPreprocessing.__init__ ->")

        
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stemmer = SnowballStemmer('spanish')
        self.stop_words = set(stopwords.words('spanish'))

    def text_preprocess(self, text):
        if not isinstance(text, str):
            return ""

        text = text.lower()

        tokens = self.tokenizer.tokenize(text)
        tokens = [w for w in tokens if w not in self.stop_words]
        tokens = [self.stemmer.stem(w) for w in tokens]


        return ' '.join(tokens)   
