import os
import numpy as np
import matplotlib.pyplot as plt
import time
import keras

#################################### Import Data ##########################################
def import_data(file):
	file_dir = os.path.dirname(__file__)
	file_path= os.path.join(file_dir,'Compressed_data/'+file)            ### Make sure the dataset is in the correct folder
	train_data= np.loadtxt(file_path, dtype= float , delimiter= ',', skiprows= 1)
	return train_data

start_time = time.time()        #### Tracking the execution time

data = import_data('gdata.csv')            #### Importing data
cross_val = 1060904              #### Row number after which the data of the last user is recorded (For the purpose of crossvalidation)
m = 1060904                      #### Training only on data of two users, you guys can check out for 8 users (set m = 993720 for adata and m=1060904 for gdata)
train_data = data[0:m,:]
test_data = data[cross_val:,:]

#Setting up the feature matrix and output vector
parameters = len(train_data[0,:]) - 2       #### Setting up the number of parameters

X_train = train_data[:,1:parameters+1]
y_train = train_data[:,parameters+1]

X_test = test_data[:,1:parameters+1]
y_test = test_data[:,parameters+1]


