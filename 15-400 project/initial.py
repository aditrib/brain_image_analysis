#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 20:24:41 2019

@author: aditribhagirath
"""

"""Initial loading of the matrix object. Note that here, the
   dimensions are 5176 X 306 X 20. The number of words is 5176, there
   are 306 indicators for each, and 20 time frames"""

import scipy.io as sio
import numpy as np
import pandas as pd

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



"""We want the next time step to be the output, given the previous time
# step, so we offset by 1"""
for i in range(1, 4001):
    y_train.append(x_train[i])
    
x_train = np.array(x_train[:4000])
y_train = np.array(y_train)
    
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

"""Adding the first LSTM layer and some Dropout regularisation
   In here, units and return_sequences is okay. However, you will need to 
   change input_shape parameter to be 20 and 306, corresponding to number
   of indicators. # of training examples does not explicitly need to be 
   entered. May want to later modify # of units. return_sequences is set
   to true IF and ONLY IF we want to add another layer. Default value for
   this parameter is false."""
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (20, 306)))
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
    