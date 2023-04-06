import pandas as pd
import numpy as np

prc = pd.read_csv('BTCUSDT_1min.csv', index_col=0)
prc.index = pd.to_datetime(prc.index)
news = pd.read_csv('news_data.csv')
time_freq = 60
ret = prc['close']/prc['open'].shift(time_freq) - 1
prc['ret'] = ret.shift(-time_freq-1)

news['datetime'] = news['date'] + news['time']
news['datetime'] = pd.to_datetime(news['datetime'], format="%Y-%m-%d%H:%M:%S")

data = pd.merge(news[['title', 'datetime']], prc['ret'], left_on='datetime', right_index=True)
data.set_index(data['datetime'], inplace=True)
data = data.sort_index(ascending=True)
split_id = round(0.8*len(data))
data_train = data[:split_id]
data_test = data[split_id:]
pos_median = data_train['ret'][data_train['ret'] > 0].median()
neg_median = data_train['ret'][data_train['ret'] < 0].median()
data_train['label'] = data_train['ret'].apply(lambda x: 1 if x > pos_median else (-1 if x < neg_median else 0))
data_test['label'] = data_test['ret'].apply(lambda x: 1 if x > pos_median else (-1 if x < neg_median else 0))
data_train.to_csv('data_train.csv')
data_test.to_csv('data_test.csv')

# data['ret'][data['ret'] > 0].median()*100
# data['ret'][data['ret'] > 0].mean()*100
# data['ret'][data['ret'] < 0].median()*100
# data['ret'][data['ret'] < 0].mean()*100

