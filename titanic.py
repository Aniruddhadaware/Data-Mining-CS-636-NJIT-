import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
test_data1 = pd.read_csv("test.csv")

train_data['Sex'] = np.where(train_data['Sex']=='female', 1, 0)
train_data['Cabin'] = (train_data['Cabin'].notnull()).astype('int')
train_data.loc[train_data.Embarked == 'S', 'Embarked'] = 0
train_data.loc[train_data.Embarked == 'C', 'Embarked'] = 1
train_data.loc[train_data.Embarked =='Q', 'Embarked'] = 2

test_data['Sex'] = np.where(test_data['Sex']=='female', 1, 0)
test_data['Cabin'] = (test_data['Cabin'].notnull()).astype('int')
test_data.loc[test_data.Embarked == 'S', 'Embarked'] = 0
test_data.loc[test_data.Embarked == 'C', 'Embarked'] = 1
test_data.loc[test_data.Embarked =='Q', 'Embarked'] = 2

sc = StandardScaler()

train_data = train_data[['Survived','Pclass','Age','Sex','SibSp','Parch','Fare','Cabin','Embarked']]
test_data = test_data[['Pclass','Age','Sex','SibSp','Parch','Fare','Cabin','Embarked']]


print(list(train_data))
train_data.fillna(train_data.mean(), inplace=True)
test_data.fillna(test_data.mean(), inplace=True)

X_train = train_data.iloc[:,1:9]   
y_train = train_data.iloc[:,0]     


X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(test_data)



classifier = Sequential()

#Input layer with 5 inputs neurons
classifier.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu', input_dim = 8))
#Hidden layer
classifier.add(Dense(output_dim = 2, init = 'uniform', activation = 'relu'))
#output layer with 1 output neuron which will predict 1 or 0
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

prediction = classifier.predict(X_test).tolist()

se = pd.Series(prediction)
test_data['prediction'] = se
test_data['prediction'] = test_data['prediction'].str.get(0)


series = []
for val in test_data.prediction:
    if val >= 0.5:
        series.append(1)
    else:
        series.append(0)

test_data1['Survived'] = series
test_data1 = test_data1[['PassengerId','Survived']]
test_data1.to_csv('test6.csv')

