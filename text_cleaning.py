import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

data_train = pd.read_csv('data_train.csv')
data_test = pd.read_csv('data_test.csv')

def text_cleaning(data):
    corpus = []
    for i in range(0, len(data)):
        text = re.sub('[^a-zA-Z]', ' ', data[i])  # replace everything that is not letters by space
        text = text.lower()  # transform capital letter to lower case
        text = text.split()
        ps = PorterStemmer()  # stemming consists of taking only the root(meaning) of a word e.g. 'loved' -> 'love'
        tile = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]  # apply stemming
        text = ' '.join(text)
        corpus.append(text)
    return corpus

text_train = text_cleaning(data_train['title'])
text_test = text_cleaning(data_test['title'])