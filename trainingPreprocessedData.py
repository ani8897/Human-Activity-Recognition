import os
import numpy as np
import pandas as pd
from numpy import genfromtxt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,f1_score
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt


X_train = genfromtxt("X_train.txt", delimiter =' ')
X_test = genfromtxt("X_test.txt", delimiter =' ')
y_test = genfromtxt("y_test.txt", delimiter ='\n')
y_train = genfromtxt("y_train.txt", delimiter ='\n')

y_train = np.array(y_train)
y_test = np.array(y_test)

print(X_train.shape)
print(y_train.shape)

def NeuralNetworkTrain(X_train,y_train,X_test,y_test):
	train_scores = np.empty(0)
	test_scores = np.empty(0)
	indices = np.empty(0)
	for i in 1,15:
		print("hidden layer: ",i,"\n")
		mlp = MLPClassifier(hidden_layer_sizes=(i, i, i))
		mlp.fit(X_train,y_train)

		predictions_train = mlp.predict(X_train)
		print("Fitting of train data for size ",i," : \n",classification_report(y_train,predictions_train))

		predictions_test = mlp.predict(X_test)
		print("Fitting of test data for size ",i," : \n",classification_report(y_test,predictions_test))

		train_scores = np.append(train_scores, f1_score(y_train,predictions_train,average='macro'))
		test_scores = np.append(test_scores, f1_score(y_test,predictions_test,average='macro'))
		indices = np.append(indices,i)

	
	plt.plot(indices, train_scores)
	plt.plot(indices,test_scores)

	plt.legend(['Train scores','Test scores'],loc='upper left')
	plt.show()
	

NeuralNetworkTrain(X_train,y_train,X_test,y_test)