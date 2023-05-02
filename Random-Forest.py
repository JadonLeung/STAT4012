#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import pandas as pd
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
data_train = pd.read_csv('https://raw.githubusercontent.com/JadonLeung/STAT4012/main/data_train.csv')
data_test = pd.read_csv('https://raw.githubusercontent.com/JadonLeung/STAT4012/main/data_test.csv')

data_train
print(len(data_train.loc[data_train['label'] == 1]))
print(len(data_train.loc[data_train['label'] == 0]))
print(len(data_train.loc[data_train['label'] == -1]))

print(len(data_test.loc[data_test['label'] == 1]))
print(len(data_test.loc[data_test['label'] == 0]))
print(len(data_test.loc[data_test['label'] == -1]))


# In[2]:


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

text_train = text_cleaning(data_train.heading)
text_test = text_cleaning(data_test.heading)
text_train


# In[3]:


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
dummy_y_train
dummy_y_test


# In[4]:


print(X_train)
y_train
data_train
en_y_train


# In[5]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[6]:


randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(X_train,y_train)

predictions = randomclassifier.predict(X_test)

matrix = confusion_matrix(data_test['label'],predictions)
print(matrix)
score = accuracy_score(data_test['label'],predictions)
print(score)
report = classification_report(data_test['label'],predictions)
print(report)


# In[7]:


import matplotlib.pyplot as plt
protfolio_ret = np.cumprod(predictions * data_test['ret'] + 1)
plt.plot(protfolio_ret)
plt.show()
max_1 = protfolio_ret.cummax()
drawdown = max_1 - protfolio_ret
max_drawdown = drawdown / max_1
print(f"Portfolio Return: {protfolio_ret.iloc[-1] - 1}")
print('Maximum Drawdown:', 100*(max_drawdown.max()), '%')
plt.plot(protfolio_ret)
plt.show()
returns = data_test['ret'] * y_pred
len(returns[returns > 0]) / (len(returns[returns > 0]) + len(returns[returns < 0]))


# In[31]:


from datetime import timedelta
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
returns = pd.DataFrame(data_test['ret'] * predictions)
returns = delete_within_24_hours(returns[returns['ret'] != 0])
portfolio_ret = np.cumprod(returns + 1)
max_1 = portfolio_ret.cummax()
drawdown = max_1 - portfolio_ret
max_drawdown = drawdown / max_1
print(f"Portfolio Return: {portfolio_ret['ret'][-1] - 1}")
print('Maximum Drawdown:', 100*(max_drawdown.max()), '%')
returns = data_test['ret'] * predictions
print(len(returns[returns > 0]) / (len(returns[returns > 0]) + len(returns[returns < 0])))

plt.plot(portfolio_ret)
plt.show()

