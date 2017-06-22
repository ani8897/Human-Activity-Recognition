import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import  sequence

start_time = time.time()					#To keep track of time to run the code

print('Loading data ...')
data1 = pd.read_csv('Phones_accelerometer.csv')		#Loading Accelerometer data
data2 = pd.read_csv('Phones_gyroscope.csv')			#Loading Gyroscope data

length1 = len(data1)
length2 = len(data2)
length = min(length1, length2)						#To make the length of the merged data equal to minimum of the two data 
# length = round(0.7*length)
data1 = data1.drop(labels = ['Arrival_Time','Creation_Time','Index', 'User'], axis=1)		#Dropping the unnecessary fields
data2 = data2.drop(labels = ['Arrival_Time','Creation_Time','Index', 'User','Model','Device'], axis=1)

data1 = data1.head(length)							#Taking only the top 'length' number of entries from both the data
data2 = data2.head(length)

data2.columns = ['x1', 'y1', 'z1', 'gt1']					#Renaming the column values of data2 as data1 would have same 'x','y' and 'z' variables
# print(data2.iloc[[9126682]])
data = pd.concat([data1, data2], axis=1)			#Merging both the accelerometer and the gyroscope data			

to_drop = ['null']									#To drop the null values fro both data1 and data2
data = data[~data['gt'].isin(to_drop)]
data = data[~data['gt1'].isin(to_drop)]

data = data.drop(labels = ['gt1'], axis=1)

data = data.iloc[::10, :]

cols_to_norm = ['x','y','z', 'x1','y1','z1']		#Normalizing the columns
data[cols_to_norm] = data[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))

y = data[['gt']]									#Extracting only the action values
data = data.drop(labels = ['gt'], axis = 1)
data = pd.get_dummies(data)							#For One Hot Encoding of the data
parameters1 = len(data.columns)	

y = pd.get_dummies(y)
parameters2 = len(y.columns)


data = np.array(data)
y = np.array(y)

m = len(data)
crossval = round(3*m/4)								#Taking 75% of the data for training and rest 25% for testing
train_data = data[0:crossval,:]
train_data_y = y[0:crossval,:]

test_data = data[crossval:,:]
test_data_y = y[crossval:,:]

X_train = train_data 								#Reshaping the data into the form required for LSTM
X_train = np.reshape(X_train, (X_train.shape[0],1,X_train.shape[1]))
X_train = np.array(X_train)

y_train = train_data_y

X_test = test_data
X_test = np.reshape(X_test, (X_test.shape[0],1,X_test.shape[1]))
X_test = np.array(X_test)

y_test = test_data_y
np.random.seed(7)

#Making the LSTM model
model = Sequential()
model.add(LSTM(24, input_dim = parameters1,return_sequences=True))
model.add(LSTM(12))
model.add(Dense(6, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

Accuracy_Arr = np.empty(0)
batch_size_Arr = np.empty(0)
#Fitting data
batch_side = 8
for i in range(5,batch_side):
	model.fit(X_train, y_train, epochs=3, batch_size=pow(2,i))
	scores = model.evaluate(X_test, y_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))
	Accuracy_Arr = np.append(Accuracy_Arr,scores[1]*100)
	batch_size_Arr = np.append(batch_size_Arr, pow(2,i))

fig = plt.figure()								#For making a plot of Accuracy vs batch size
plt.plot(batch_size_Arr, Accuracy_Arr)
plt.xlabel("Batch size")
plt.ylabel("Accuracy of Model(in %)")
plt.show()
fig.save("Merged_data_Accuracy_vs_batch_size.png")

model.save("my_model.h5")						#For saving the model