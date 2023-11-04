import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


prc = pd.read_csv('BTCUSDT_1min_2.csv', index_col=0)
prc.index = pd.to_datetime(prc.index)
news = pd.read_csv('coindesk_data_from_refinitiv.csv')
time_freq = 15
ret = prc['close']/prc['open'].shift(time_freq) - 1
prc['ret'] = ret.shift(-time_freq-1)

news['datetime'] = news['date'] + news['time']
news['datetime'] = pd.to_datetime(news['datetime'], format="%Y-%m-%d%H:%M:%S").apply(lambda x: x.replace(second=0))

data = pd.merge(news[['headline', 'datetime']], prc['ret'], left_on='datetime', right_index=True)
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

data_train['label'] = data_train['ret'].apply(lambda x: 1 if x > 0 else 0)
data_test['label'] = data_test['ret'].apply(lambda x: 1 if x > 0 else 0)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.callbacks import EarlyStopping

X_train = data_train.iloc[:, 3: -1].values
X_test = data_test.iloc[:, 3: -1].values
y_train = data_train.iloc[:, -1].values
y_test = data_test.iloc[:, -1].values
# le = LabelEncoder()
# le.fit_transform(y_train)
# en_y_train = le.transform(y_train)
# en_y_test = le.transform(y_test)
# dummy_y_train = np_utils.to_categorical(en_y_train)
# dummy_y_test = np_utils.to_categorical(en_y_test)

model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Embedding(1500, 45, input_length=X_train.shape[1]))
# model.add(tf.keras.layers.Conv1D(128, 5, activation='relu'))
# model.add(tf.keras.layers.GlobalMaxPool1D())
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=100)
# model.fit(X_train, y_train, validation_split=0.2, batch_size=32, epochs=100, callbacks=[EarlyStopping(monitor='val_loss', mode='min', verbose=1)])

y_pred = model.predict(X_test)
y_pred = np.where(y_pred > 0.5, 1, 0)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy score: {accuracy}")
protfolio_ret = np.cumprod(data_test['ret'] * y_pred.squeeze() + 1)
print(f"Portfolio return: {protfolio_ret[-1] - 1}")
plt.plot(protfolio_ret)
plt.show()

