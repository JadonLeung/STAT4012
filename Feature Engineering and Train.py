import pandas as pd

train_data = pd.read_csv('https://raw.githubusercontent.com/JadonLeung/STAT4012/main/data_train.csv')
test_data = pd.read_csv('https://raw.githubusercontent.com/JadonLeung/STAT4012/main/data_test.csv')

data = train_data.title
data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)

testdata = test_data.title
testdata.replace("[^a-zA-Z]"," ",regex=True, inplace=True)

list1= [i for i in range(932)]
new_Index=[str(i) for i in list1]
data.columns=new_Index
data.head(5)

list2= [i for i in range(233)]
new_Index=[str(i) for i in list2]
testdata.columns=new_Index
testdata.head(5)

train_dataset = data.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

for i,row in data.items(): 
    data.iloc[i]=data.iloc[i].lower()

for i,row in testdata.items(): 
    testdata.iloc[i]=testdata.iloc[i].lower()
    
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

countvector=CountVectorizer(ngram_range=(2,2))
traindataset=countvector.fit_transform(data)

randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(traindataset,train_data['label'])

test_transform= []
test_data1 = countvector.transform(testdata)
predictions = randomclassifier.predict(test_data1)

matrix = confusion_matrix(test_data['label'],predictions)
print(matrix)
score = accuracy_score(test_data['label'],predictions)
print(score)
report = classification_report(test_data['label'],predictions)
print(report)

#accuracy 0.54
