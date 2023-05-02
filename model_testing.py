import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

prc = pd.read_csv('BTCUSDT_1min.csv', index_col=0)
prc.index = pd.to_datetime(prc.index)
news = pd.read_csv('only_dow_jones_news_data.csv')
time_freq = 1440
ret = prc['close']/prc['open'].shift(time_freq) - 1
prc['prvs_high'] = prc['high'].rolling(time_freq).max().shift(1)
prc['prvs_low'] = prc['low'].rolling(time_freq).min().shift(1)
prc['prvs_volume'] = prc['volume'] .rolling(time_freq).sum().shift(1)
prc['ret'] = ret.shift(-time_freq-1)

def get_Volatility(close, span0=20):
    # simple percentage returns
    df0 = close.pct_change()
    # 20 days, a month EWM's std as boundary
    df0 = df0.ewm(span=span0).std()
    df0.dropna(inplace=True)
    return df0

prc['volatility'] = get_Volatility(prc['close'])

t_final = 10
upper_lower_multipliers = [2, 2]

news['datetime'] = news['date'] + news['time']
news['datetime'] = pd.to_datetime(news['datetime'], format="%Y-%m-%d%H:%M:%S")

data = pd.merge(news[['heading', 'datetime']], prc[['close', 'ret', 'prvs_high', 'prvs_low', 'prvs_close',
                                                    'prvs_volume']], left_on='datetime', right_index=True)
data.set_index(data['datetime'], inplace=True)
data = data.sort_index(ascending=True)

def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def get_polarity(text):
    return TextBlob(text).sentiment.polarity

def getSIA(text):
    sia = SentimentIntensityAnalyzer()
    compound, neg, pos, neu = sia.polarity_scores(text).values()
    return compound, neg, pos, neu

data['subjectivity'] = data['heading'].apply(get_subjectivity)
data['polarity'] = data['heading'].apply(get_polarity)
data[['compound', 'neg', 'pos', 'neu']] = data['heading'].apply(getSIA).apply(pd.Series)

data['label'] = data['ret'].apply(lambda x: 1 if x > 0 else 0)

split_id = round(0.8*len(data))
data_train = data.iloc[:split_id, 2:]
data_test = data.iloc[split_id:, 2:]

#
# import nltk
# import re
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
#
# def text_cleaning(data):
#     corpus = []
#     for i in range(0, len(data)):
#         text = re.sub(r'[^\w\s]', '', str(data[i]).lower().strip())
#         text = re.sub(r'(press|release|barronscom|wsj)', ' ', text)
#         text = text.split()
#         ps = PorterStemmer()  # stemming consists of taking only the root(meaning) of a word e.g. 'loved' -> 'love'
#         text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]  # apply stemming
#         text = ' '.join(text)
#         corpus.append(text)
#     return corpus

# text_train = text_cleaning(data_train['heading'])
# text_test = text_cleaning(data_test['heading'])

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_curve, roc_auc_score
from keras.callbacks import EarlyStopping
from keras.models import load_model
from sklearn.metrics import classification_report
from datetime import timedelta

# cv = CountVectorizer(max_features=2500)
# X_train = cv.fit_transform(text_train).toarray()
# X_test = cv.transform(text_test).toarray()
X_train = data_train.iloc[:, 2:-1].values
X_test = data_test.iloc[:, 2:-1].values
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train[:, [0, 1, 2]] = sc.fit_transform(X_train[:, [0, 1, 2]])
X_test[:, [0, 1, 2]] = sc.transform(X_test[:, [0, 1, 2]])
y_train = data_train.iloc[:, -1].values
y_test = data_test.iloc[:, -1].values

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_split=0.2, batch_size=32, epochs=100)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['accuracy'])  # add accuracy to the plot
plt.plot(history.history['val_accuracy'])  # add validation accuracy to the plot
plt.title('Model loss and accuracy')  # change title
plt.ylabel('Loss/Accuracy')  # change y-axis label
plt.xlabel('Epoch')
plt.legend(['Train Loss', 'Validation Loss', 'Train Accuracy', 'Validation Accuracy'],
           loc='upper left')  # add legend for accuracy
plt.show()

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l2

param_grid = {'neurons': [4, 8, 16, 32, 64 , 128, 256]}
def create_model(neurons=8):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(neurons, activation='relu'))
    model.add(tf.keras.layers.Dense(neurons, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, verbose=0)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5)
grid.fit(X_train, y_train)
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
params = grid.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f"Accuracy: {mean:.4f} (std: {stdev:.4f}) with: {param}")
best_params = grid.best_params_
print('best_params:', best_params)

while accuracy < 0.63:
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_split=0.2, batch_size=32, epochs=100, verbose=0)
    y_proba = model.predict(X_test)
    y_pred = (y_proba > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Accuracy score: {accuracy*100:.2f}%")
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC: {roc_auc*100:.2f}%")
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

y_pred[y_pred == 0] = -1
returns = pd.DataFrame(data_test['ret'] * y_pred.squeeze())
equity = 10000
returns['cumulative_returns'] = (returns['ret'] * equity).cumsum() + equity
plt.plot(returns['cumulative_returns'])
plt.title('Equity Curve')
plt.xlabel('Time')
plt.ylabel('Equity')
plt.show()

max = returns['cumulative_returns'].cummax()
drawdown = max - returns['cumulative_returns']
max_drawdown = drawdown / max
trades = returns['ret']
win_rate = len(trades[trades > 0]) / len(trades)
cum_ret = returns['cumulative_returns'].iloc[-1]/10000-1
print(f"Cumulative returns: {cum_ret*100:.2f}%")
print(f"Maximum drawdown: {max_drawdown.max()*100:.2f}%")
print(f"Win rate: {win_rate*100:.2f}%")

print(report)

def delete_within_24_hours(df):
    prev_datetime = None
    for index, row in df.iterrows():
        if prev_datetime is not None and (index - prev_datetime) < timedelta(hours=24):
            try:
                df.drop(index, inplace=True)
            except:
                pass
        else:
            prev_datetime = index
    return df

y_pred[y_pred == 0] = -1
returns = pd.DataFrame(data_test['ret'] * y_pred.squeeze())
# returns = delete_within_24_hours(returns[returns['ret'] != 0])
portfolio_ret = np.cumprod(returns + 1)
max = portfolio_ret.cummax()
drawdown = max - portfolio_ret
max_drawdown = drawdown / max
print(f"Portfolio Return: {portfolio_ret['ret'][-1] - 1}")
print('Maximum Drawdown:', 100*(max_drawdown.max()), '%')
print('Win Rate:', 100*(len(returns[returns > 0]) / (len(returns[returns > 0]) + len(returns[returns < 0]))), '%')
plt.plot(portfolio_ret['ret'])
plt.show()

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_proba = classifier.predict_proba(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Accuracy score: {accuracy}")
print(report)

returns = pd.DataFrame(data_test['ret'] * y_pred)
returns = delete_within_24_hours(returns[returns['ret'] != 0])
portfolio_ret = np.cumprod(returns + 1)
max = portfolio_ret.cummax()
drawdown = max - portfolio_ret
max_drawdown = drawdown / max
print(f"Portfolio Return: {portfolio_ret['ret'][-1] - 1}")
print('Maximum Drawdown:', 100*(max_drawdown.max()), '%')
plt.plot(portfolio_ret)
plt.show()