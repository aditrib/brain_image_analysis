#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 23:11:30 2019

@author: aditribhagirath
"""


"""Initial loading of the matrix object. Note that here, the
   dimensions are 5176 X 306 X 20. The number of words is 5176, there
   are 306 indicators for each, and 20 time frames"""

import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle

mat = sio.loadmat(("B_HP_notDetrended_25ms.mat"))
#print(mat)
data = mat['data'].astype('float32')

cleanData = []

for i in range(len(data)):
    currWord = data[i]
    newWord = currWord.transpose()
    cleanData.append(newWord)
    
""" Proper data, shaped as desired. Here, the matrix dimensions are 
    5176 X 20 X 306 """
    
cleanNpData = np.array(cleanData)

# Simple test comparision of the first word in modified and 
# unmodified data
first1 = data[0]
first2 = cleanNpData[0]


x_train = cleanData[:4001]
y_train = []

TwoDSeries = []

for word in x_train:
    wordTime = len(word) 
    for j in range(wordTime):
        TwoDSeries.append(word[j])
        

    


"""We want the next time step to be the output, given the previous time
# step, so we offset by 1"""
for i in range(1, 80020):
    y_train.append(TwoDSeries[i])
    
x_train = TwoDSeries[:80019]

    
x_train = np.array(x_train)
y_train = np.array(y_train)
    
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten

# Need to reshape the array to make it three dimensional again. 
# It is now with 80019 rows, each of which is 1 X 306 array. 

x_train = np.reshape(x_train,(x_train.shape[0], 1, x_train.shape[1]))
y_train = np.reshape(y_train,(y_train.shape[0], 1, y_train.shape[1]))

# Initialising the RNN
regressor = Sequential()

"""Adding the first LSTM layer and some Dropout regularisation
   Here, the input shape has been changed to 1 X 306, because each training
   sample has 1 row, 306 cols. # of training examples does not explicitly need to be 
   entered. May want to later modify # of units. return_sequences is set
   to true IF and ONLY IF we want to add another layer. Default value for
   this parameter is false."""
#regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (1, 306)))
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (1, 306)))
# 20% of neurons of LSTM will be ignored during the training. This is 
# dropout regularization.
#regressor.add(Dropout(0.2))    

# Second layer. Do not need to specify input shape here.
regressor.add(LSTM(units = 50, return_sequences = True))

regressor.add(LSTM(units = 50, return_sequences = True))


# Adding the output layer. In our case, the number of units is 306, which is 
# equal to number of indicators.
regressor.add(Dense(units = 306))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(x_train, y_train, epochs = 100, batch_size = 32)


# Saving model so we don't have to keep re-training it

filename = 'savemodel.sav'
pickle.dump(regressor, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))


# Getting the real stock price of 2017

x_testing = cleanData[4010:5000]

x_test_final = []

for word in x_testing:
    wordTime = len(word) 
    for j in range(wordTime):
        x_test_final.append(word[j])
# Getting the predicted stock price of 2017

X_test = np.array(x_test_final)
X_test = np.reshape(X_test,(X_test.shape[0], 1, X_test.shape[1]))
predicted_brain_image = loaded_model.predict(X_test)

# Visualising the results of the data that has been created

first_real = []

for i in range(0, len(X_test)):
    current = X_test[i]
    first_real.append(current[0][175])
    
first_predicted = []

for i in range(0, len(X_test)):
    current = predicted_brain_image[i]
    first_predicted.append(current[0][175])


# Visualising the results
first_real = first_real[:20]
first_predicted = first_predicted[:20]
    
plt.plot(first_real, color = 'red', label = 'Real First Indicator')
plt.plot(first_predicted, color = 'blue', label = 'Predicted First Indicator')
plt.title('First Indicator Pred. Evaluation')
plt.xlabel('Time')
plt.ylabel('First Indicator')
plt.legend()
plt.show()

import scipy
from scipy import stats


def calcCorr(M1, M2):
    z1 = scipy.stats.zscore(M1, 0)
    z2 = scipy.stats.zscore(M2, 0)
    n = len(M1)
    return (1/n)*(np.sum(z1*z2, 0))
    
correlations = calcCorr(X_test,  predicted_brain_image)



from sklearn.metrics import mean_squared_error

    
plt.plot(correlations.transpose(), color = 'red', label = 'Corr')

plt.title('First Indicator Pred. Evaluation')
plt.xlabel('Time')
plt.ylabel('First Indicator')
plt.legend()
plt.show()



# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()




