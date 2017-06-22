import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from keras.models import Sequential
from keras.layers import Dense,LSTM,GRU,Dropout, Flatten

#################################### Import Data ##########################################
def import_data(file):
	file_dir = os.path.dirname(__file__)
	file_path= os.path.join(file_dir,'Compressed_data/'+file)            ### Make sure the dataset is in the correct folder
	train_data= np.loadtxt(file_path, dtype= float , delimiter= ',', skiprows= 1)
	return train_data

def normalize(data_vector):
	max_data = np.amax(data_vector)
	min_data = np.amin(data_vector)
	data_vector = -1 + 2*(data_vector -min_data)/(max_data - min_data)
	return data_vector

#### Tracking the execution time
start_time = time.time()

#### Importing data
print('Loading data ...')
data = import_data('adata.csv')

m = len(data)
crossval = round(3*m/4)
train_data = data[0:crossval,:]
test_data = data[crossval:,:]

#### Setting up the number of parameters
parameters = len(train_data[0,:]) - 2

#### Normalizing the data
for x in range(1,parameters+1):
	train_data[:,x] = normalize(train_data[:,x])
	test_data[:,x] = normalize(test_data[:,x])



X_train = train_data[:,1:parameters+1]

X_train = np.reshape(X_train, (X_train.shape[0],1,X_train.shape[1]))
X_train = np.array(X_train)

y_train = train_data[:,parameters+1]
y_train = pd.get_dummies(y_train)
y_train = np.array(y_train)

X_test = test_data[:,1:parameters+1]
X_test = np.reshape(X_test, (X_test.shape[0],1,X_test.shape[1]))
X_test = np.array(X_test)

y_test = test_data[:,parameters+1]
y_test = pd.get_dummies(y_test)
y_test = np.array(y_test)

np.random.seed(7)

AccuracyArr = np.empty(0)
sArr = np.empty(0)
for s in 6,12,18:
	model = Sequential()
	model.add(LSTM(s, input_shape=(None, parameters), return_sequences= False))
	#model.add(Dropout(0.9))
	#model.add(LSTM(18, return_sequences=True))
	#model.add(Dropout(0.2))
	#model.add(LSTM(12))
	model.add(Dense(6, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	print(model.summary())
	model.fit(X_train, y_train, epochs=2, batch_size=40)
	# Final evaluation of the model
	scores = model.evaluate(X_test, y_test, verbose=0)
	AccuracyArr = np.append(AccuracyArr,(scores[1]*100))
	sArr = np.append(sArr,s)
	print("Accuracy: %.2f%%" % (scores[1]*100))

fig = plt.figure()
plt.plot(sArr,AccuracyArr)
plt.xlabel('#LSTM neurons')
plt.ylabel('Accuracy of model (in %)')
plt.show()
fig.savefig('Accelerometer.png')
