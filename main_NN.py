import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

##################
from sklearn.ensemble import RandomForestClassifier
###################

#################################### Import Data ##########################################
def import_data(file):
    file_dir = os.path.dirname(__file__)
    file_path= os.path.join(file_dir,'Compressed_data/'+file)            ### Make sure the dataset is in the correct folder
    train_data= np.loadtxt(file_path, dtype= float , delimiter= ',', skiprows= 1)
    return train_data

############################################## Neural Network Implementation ####################################################
def NeuralNetworkTrain(X_train,y_train,X_test,y_test):
    train_scores = np.empty(0)
    test_scores = np.empty(0)
    indices = np.empty(0)
    for i in 5, 10, 15:                    #### We are taking only one hidden layer, try with different number of layers
        print("hidden layer: ",i,"\n")
        mlp = MLPClassifier(hidden_layer_sizes=(i,i,i),early_stopping=True,learning_rate='adaptive',learning_rate_init=0.003)
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
    

###################################################################################################


######################################### Random Forest Implementation ##############################################
def RandomForestTrain(X_train,y_train,X_test,y_test):
    train_scores = np.empty(0)
    test_scores = np.empty(0)
    indices = np.empty(0)
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train,y_train)
    predictions_train = rf.predict(X_train)
    print("Fitting of train data : \n",classification_report(y_train,predictions_train))

    predictions_test = rf.predict(X_test)
    print("Fitting of test data for size : \n",classification_report(y_test,predictions_test))

    #train_scores = np.append(train_scores, f1_score(y_train,predictions_train,average='macro'))
    #test_scores = np.append(test_scores, f1_score(y_test,predictions_test,average='macro'))
    #indices = np.append(indices,i)

    '''
    plt.plot(indices, train_scores)
    plt.plot(indices,test_scores)

    plt.legend(['Train scores','Test scores'],loc='upper left')
    plt.show()
    '''

######################################################################################################


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


NeuralNetworkTrain(X_train,y_train,X_test,y_test)

data = import_data('adata.csv')            #### Importing data
cross_val = 993720              #### Row number after which the data of the last user is recorded (For the purpose of crossvalidation)
m = 993720                      #### Training only on data of two users, you guys can check out for 8 users (set m = 993720 for adata and m=1060904 for gdata)
train_data = data[0:m,:]
test_data = data[cross_val:,:]

#Setting up the feature matrix and output vector
parameters = len(train_data[0,:]) - 2       #### Setting up the number of parameters

X_train = train_data[:,1:parameters+1]
y_train = train_data[:,parameters+1]

X_test = test_data[:,1:parameters+1]
y_test = test_data[:,parameters+1]

NeuralNetworkTrain(X_train,y_train,X_test,y_test)
print ("time elapsed: ", format(time.time() - start_time)) #### This will take 6-7 minutes if you take the entire dataset
#RandomForestTrain(X_train,y_train,X_test,y_test)
#print ("time elapsed: ", format(time.time() - start_time)) #### This is going to take a lot of time maybe half an hour


