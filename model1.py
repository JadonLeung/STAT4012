from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

cv = CountVectorizer(max_features = 1500)
x_train = cv.fit_transform(text_train).toarray()
x_test = cv.transform(text_test).toarray()
y_train = data_train.iloc[:,-1].values
y_test = data_test.iloc[:,-1].values
le = LabelEncoder()
le.fit_transform(y_train)
en_y_train = le.transform(y_train)
en_y_test = le.transform(y_test)
dummy_y_train = np_utils.to_categorical(en_y_train)
dummy_y_test = np_utils.to_categorical(en_y_test)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, dummy_y_train, batch_size=5, epochs=200)

y_pred = np.argmax(model.predict(x_test), axis=1)
print(np.concatenate((y_pred.reshape(len(y_pred),1), en_y_test.reshape(len(y_test),1)),1))
cm = confusion_matrix(en_y_test, y_pred)
print(cm)
accuracy_score(en_y_test, y_pred)
#  accuracy_score= 0.45064377682403434
