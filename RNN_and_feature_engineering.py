import pandas as pd
import numpy as np

train_data = pd.read_csv('https://raw.githubusercontent.com/JadonLeung/STAT4012/main/data_train.csv')
test_data = pd.read_csv('https://raw.githubusercontent.com/JadonLeung/STAT4012/main/data_test.csv')

data = train_data.title
data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)

testdata = test_data.title
testdata.replace("[^a-zA-Z]"," ",regex=True, inplace=True)

train_data['word_length'] = train_data['title'].apply(lambda x:len(x.split()))
train_data

for i,row in data.items(): 
    data.iloc[i]=data.iloc[i].lower()

for i,row in testdata.items(): 
    testdata.iloc[i]=testdata.iloc[i].lower()

def preprocessingData(preprocessData):
    list1= [i for i in preprocessData]
    new_Index=[str(i) for i in list1]
    preprocessData.columns=new_Index
    for i,row in preprocessData.items(): 
        preprocessData.iloc[i]=preprocessData.iloc[i].lower()
        
    return preprocessData
    
data = preprocessingData(data)
testdata = preprocessingData(testdata)

data

from keras.preprocessing.text import Tokenizer
num_words = 10000 # this means 15000 unique words can be taken 
tokenizer=Tokenizer(num_words,lower=True)
df_total = pd.concat([data,testdata], axis = 0)
tokenizer.fit_on_texts(data)

len(tokenizer.word_index)
len(testdata.max())

from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, LSTM, Embedding,Bidirectional
import tensorflow
from tensorflow.compat.v1.keras.layers import CuDNNLSTM,CuDNNGRU
from tensorflow.keras.layers import Dropout

X_train_ =tokenizer.texts_to_sequences(data)
X_train_pad=pad_sequences(X_train_,maxlen=41,padding='post')
X_test_ = tokenizer.texts_to_sequences(testdata)
X_test_pad = pad_sequences(X_test_, maxlen = 41, padding = 'post')


import tensorflow as tf
EMBEDDING_DIM = 100 # this means the embedding layer will create  a vector in 100 dimension
model = Sequential()
model.add(Embedding(input_dim = num_words,# the whole vocabulary size 
                          output_dim = EMBEDDING_DIM, # vector space dimension
                          input_length= X_train_pad.shape[1] # max_len of text sequence
                          ))
model.add(Dropout(0.2))
model.add(Bidirectional(CuDNNLSTM(100,return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(CuDNNLSTM(200,return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(CuDNNLSTM(100,return_sequences=False)))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = tf.keras.optimizers.legacy.Adam(lr=1e-5),metrics = 'accuracy')

from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor = 'accuracy', mode = 'min', verbose = 1, patience = 200)
mc = ModelCheckpoint('./model.h5', monitor = 'val_accuracy', mode = 'max', verbose = 1, save_best_only = True)

history_embedding = model.fit(X_train_pad, train_data.label, validation_data=(X_test_pad, test_data.label), epochs = 100, batch_size = 120, verbose = 1, callbacks= [es, mc]  )

from keras.models import load_model  
new_model = load_model(r'./model.h5')
accuracy=new_model.evaluate(X_test_pad, test_data.label)[1]

accuracy: 0.5579
