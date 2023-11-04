import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

prc = pd.read_csv('BTCUSDT_1min.csv', index_col=0)
prc.index = pd.to_datetime(prc.index)
news = pd.read_csv('only_dow_jones_news_data.csv')
time_freq = 60*24
ret = prc['close']/prc['open'].shift(time_freq) - 1
prc['ret'] = ret.shift(-time_freq-1)

news['datetime'] = news['date'] + news['time']
news['datetime'] = pd.to_datetime(news['datetime'], format="%Y-%m-%d%H:%M:%S")

data = pd.merge(news[['heading', 'datetime']], prc['ret'], left_on='datetime', right_index=True)
data.set_index(data['datetime'], inplace=True)
data = data.sort_index(ascending=True)
split_id = round(0.8*len(data))
data_train = data[:split_id]
data_test = data[split_id:]

quantile = 0.35
pos_median = data_train['ret'][data_train['ret'] > 0].quantile(quantile)
neg_median = data_train['ret'][data_train['ret'] < 0].quantile(1 - quantile)
data_train['label'] = data_train['ret'].apply(lambda x: 1 if x > pos_median else (-1 if x < neg_median else 0))
data_test['label'] = data_test['ret'].apply(lambda x: 1 if x > pos_median else (-1 if x < neg_median else 0))

import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def text_cleaning(data):
    corpus = []
    for i in range(0, len(data)):
        text = re.sub(r'[^\w\s]', '', str(data[i]).lower().strip())
        text = re.sub(r'(press|release|barronscom|wsj)', ' ', text)
        text = text.split()
        ps = PorterStemmer()  # stemming consists of taking only the root(meaning) of a word e.g. 'loved' -> 'love'
        text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]  # apply stemming
        text = ' '.join(text)
        corpus.append(text)
    return corpus

words = data_train.heading.str.split()
value_counts = pd.Series([word for sentence in words for word in sentence]).value_counts()
to_remove = value_counts[value_counts > 100].index
words = words.apply(lambda sentence: [word for word in sentence if word not in to_remove])
data_train.heading = words.str.join(' ')
words = data_test.heading.str.split()
words = words.apply(lambda sentence: [word for word in sentence if word not in to_remove])
data_test.heading = words.str.join(' ')

text_train = text_cleaning(data_train['heading'])
text_test = text_cleaning(data_test['heading'])

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.callbacks import EarlyStopping

cv = CountVectorizer(max_features=2000)
X_train = cv.fit_transform(text_train).toarray()
X_test = cv.transform(text_test).toarray()
y_train = data_train.iloc[:, -1].values
y_test = data_test.iloc[:, -1].values
le = LabelEncoder()
le.fit_transform(y_train)
en_y_train = le.transform(y_train)
en_y_test = le.transform(y_test)
dummy_y_train = np_utils.to_categorical(en_y_train)
dummy_y_test = np_utils.to_categorical(en_y_test)

model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Embedding(1500, 45, input_length=X_train.shape[1]))
# model.add(tf.keras.layers.Conv1D(128, 5, activation='relu'))
# model.add(tf.keras.layers.GlobalMaxPool1D())
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, dummy_y_train, validation_data=(X_test, dummy_y_test), batch_size=32, epochs=100)
# model.fit(X_train, dummy_y_train, validation_data=(X_test, dummy_y_test), batch_size=32, epochs=100, callbacks=[EarlyStopping(monitor='val_loss', mode='min', verbose=1)])

y_pred = np.argmax(model.predict(X_test), axis=1)
print(np.concatenate((y_pred.reshape(len(y_pred),1), en_y_test.reshape(len(y_test),1)),1))
cm = confusion_matrix(en_y_test, y_pred)
print(cm)
accuracy = accuracy_score(en_y_test, y_pred)
print(f"Accuracy score: {accuracy}")
protfolio_ret = np.cumprod(data_test['ret'] * le.inverse_transform(y_pred) + 1)
print(f"Portfolio return: {protfolio_ret[-1] - 1}")
plt.plot(protfolio_ret)
plt.show()

