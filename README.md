# Human-Activity-Recognition

## Dataset link :
https://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition

## Libraries used :
Keras, Scikit-Learn, Numpy, Matplotlib and Pandas

## File Structure: 

There are 8 main files: 4 for data management, 4 for Machine learning codes and 1 for plotting the results.

### Data management files: 
1.As the dataset was very huge (~ 1.4 GB), it was partitioned into 13 files and the scripts 'compress_file.py' and 'compress2.0.py' were used to downsample the dataset stored in these 13 files to obtain 13 compressed files. 

2.The scripts 'merge.py' and 'merge2.0.py' are used to merge the compressed files to obtain the dataset which was used for training. The 2.0 scripts were used for merging the accelerometer and gyroscope data.

### Machine Learning codes: 
1."main_NN.py" contains the Neural network implementation which was used on the accelerometer and gyroscope data separately. 

2."main_RNN.py" contains the LSTM implementation which used on the merged data as well as the accelerometer and gyroscope data separately. 

3."main.py" takes in the complete dataset (not the compressed dataset) and implements LSTM. 

4."trainingPreprocessedData.py" takes in the dataset (Link:- https://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions) and outputs the result, this file was mainly created to see whether our LSTM model was good enough(The accuracy obtained from this preprocessed dataset was 91%).
### Plotting:
Used for plotting the results obtained from "main_NN.py".

## Model:
"model.h5" stores the final model to the problem.

## NOTE:
Final code is run by 'main.py' and for this the dataset must be in the same folder and run the script using python3  




