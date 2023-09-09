#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 22:15:18 2019

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
   
import scipy
from scipy import stats

import mne # needed for the topoplot
import csv

with open('locations.txt', 'r') as f:
    locs = csv.reader(f,delimiter=',')
    loc306 = np.array([[float(w1[0].split(' ')[1]),float(w1[0].split(' ')[2])] for w1 in locs ])







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
regressor.add(LSTM(units = 50, return_sequences = True, 
                   batch_input_shape = (1, 1, 306), stateful = True))
# 20% of neurons of LSTM will be ignored during the training. This is 
# dropout regularization.
#regressor.add(Dropout(0.2))    

# Second layer. Do not need to specify input shape here.
regressor.add(LSTM(units = 50, return_sequences = True, stateful = True))

regressor.add(LSTM(units = 50, return_sequences = True, stateful = True))


# Adding the output layer. In our case, the number of units is 306, which is 
# equal to number of indicators.
regressor.add(Dense(units = 306))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
# Batch size must be a multiple of number of training instances. In this case, 80019.
# Fitting the RNN to the Training set
"""for i in range(10):
    regressor.fit(x_train, y_train, epochs = 1, batch_size = 1, shuffle = False)
    regressor.reset_states()
"""

# Saving model so we don't have to keep re-training it

filename = 'statefulmodel10iter.sav'
#pickle.dump(regressor, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))


# Getting the real stock price of 2017
# x_testing is a list of 990 samples. Each of these samples has dimensions 
# 20 X 306. We need to loop through each of the 20, and make a list with
# 990 X 20 elements = 19800
x_testing = cleanData[4010:5000]

x_test_final = []

for word in x_testing:
    wordTime = len(word) 
    for j in range(wordTime):
        x_test_final.append(word[j])
        
# Getting the predictions all in one go. Predicted_brain_image has dimensions
# The same as x_test (19800 X 1 X 306). 1 is only to make it the correct 
# shape for the purpose of predictions

X_test = np.array(x_test_final[:16000])
X_test = np.reshape(X_test,(X_test.shape[0], 1, X_test.shape[1]))
#predicted_brain_image = loaded_model.predict(X_test)

firstBrainScan = X_test[0]
firstBrainShaped = np.reshape(firstBrainScan,(1, 1, 306))
firstPred = loaded_model.predict(firstBrainShaped)

h = mne.viz.plot_topomap(firstPred.flatten(),loc306, vmin = -2 , vmax = 2 )
#topoplot(firstPred)

# subset has dimensions #rows of dataset X 1 X 306
# First, make predictions for the first 100 points. Then, starting from the
# 101st point, start a chain of 3 predictions
"""
def makePredictions(subset, numIter):
    all = []
    for i in range(100):
        current = subset[i]
        shapedCurrent = np.reshape(current, (1, 1, 306))
        loaded_model.predict(shapedCurrent)
    rest = subset[100:]
    for indicator in rest:
        pt = indicator
        shapedPt = np.reshape(pt,(1, 1, 306))
        for i in range(numIter):
            predTPlus1 = loaded_model.predict(shapedPt)
            shapedPt = predTPlus1
        all.append(shapedPt)
    return all
"""

# This function takes in a given dataset, the start index for this dataset where
# We'd like to start making our predictions, the number of previous steps we
# need to predict for maintaining the state, the number of forward predictions
# we would like to make (numIter) and the range over which we would like to 
# make these predictions, starting from startI.


def predictWithStateful(subset, startI, prevSteps, numIter, endI):
    all = []
    for i in range(startI, endI):
        for j in range(i-prevSteps, i):
            current = subset[j]
            shapedCurrent = np.reshape(current, (1, 1, 306))
            loaded_model.predict(shapedCurrent)
        stept = subset[i]
        shapedStept = np.reshape(stept, (1, 1, 306))
        for k in range(numIter):
            stepTPlus1 = loaded_model.predict(shapedStept);
            shapedStept = stepTPlus1
        all.append(shapedStept)
    return all
            


#finalResults = makePredictions(smallSubset, 3)

#pt = smallSubset[0]sss
#shapedPt = np.reshape(pt,(1, 1, 306))
#predTPlus1 = loaded_model.predict(shapedPt)

def getIndicator(subset, index):
    indicators = []
    for i in range(len(subset)):
        indicators.append(subset[i][0][index])
    return indicators

def getIndicatorPred(subset, index):
    indicators = []
    for i in range(len(subset)):
        indicators.append(subset[i][0][0][index])
    return indicators
     
def calcCorr(M1, M2):
    z1 = scipy.stats.zscore(M1, 0)
    z2 = scipy.stats.zscore(M2, 0)
    n = len(M1)
    return (1/n)*(np.sum(z1*z2, 0))
    

correlationSum = [0 for i in range(306)]

import numpy

"""
def plotCorrelation(indicatorNum, lo, hi):
    smallSubset = X_test[:200]
    realValues = X_test[101: 201]
    # subset, start index, previous steps, previous points, timesteps, end index
    finalRes = predictWithStateful(smallSubset, 100, 100, 1, 200)
    
    
    
    realIndic = getIndicator(realValues[lo:hi], indicatorNum)
    predIndic = scipy.stats.zscore(getIndicatorPred(finalRes[lo:hi], indicatorNum))
    #correlations = calcCorr(realValues,  finalRes)
    #correlSum = sum(correlations)
    #correlationSum[indicatorNum] = correlSum
    
    
    allPred = scipy.stats.zscore(getIndicatorPred(finalRes, indicatorNum))
    allReal = getIndicator(realValues, indicatorNum)
    print(numpy.corrcoef(allReal,allPred))
    
    plt.plot(realIndic, color = 'red', label = 'Real First Indicator')
    plt.plot(predIndic, color = 'blue', label = 'Predicted First Indicator')
    plt.title(str(indicatorNum) + 'Indicator Pred. Evaluation')
    plt.xlabel('Time')
    plt.ylabel('First Indicator')
    plt.legend()
    plt.show()
"""

def findWordCorrelations(L1, L2):
    correlations = []
    set1 = []
    set2 = []
    for i in range(10):
        set1 = []
        set2 = []
        for j in range(120 + i, len(L1), 20):
            set1.append(L1[j])
            set2.append(L2[j])
        correlation = numpy.corrcoef(set1, set2)
        correlations.append(numpy.ndarray.tolist(correlation)[1][0])
    print(correlations)

    
def plotCorrelation(indicatorNum, lo, hi):
    smallSubset = X_test[:500]
    realValues = X_test[101: 501]
    # subset, start index, previous steps, previous points, timesteps, end index
    finalRes = predictWithStateful(smallSubset, 100, 100, 1, 500)
    
    
    
    realIndic = getIndicator(realValues[lo:hi], indicatorNum)
    predIndic = scipy.stats.zscore(getIndicatorPred(finalRes[lo:hi], indicatorNum))
    #correlations = calcCorr(realValues,  finalRes)
    #correlSum = sum(correlations)
    #correlationSum[indicatorNum] = correlSum
    
    
    allPred = scipy.stats.zscore(getIndicatorPred(finalRes, indicatorNum))
    allReal = getIndicator(realValues, indicatorNum)
    print(numpy.ndarray.tolist(numpy.corrcoef(allReal,allPred))[1][0])
    findWordCorrelations(allReal, allPred)
    
    plt.plot(realIndic, color = 'red', label = 'Real Sensor Value')
    plt.plot(predIndic, color = 'blue', label = 'Predicted Sensor Value')
    plt.title(str(indicatorNum) + ' Indicator Pred. Evaluation')
    plt.xlabel('Time')
    plt.ylabel('Sensor Value ')
    plt.legend()
    plt.show()
    
plt.plot([0.40, 0.06, 0.28, -0.01, 0.39, 0.15, 0.01, 0.21, 0.63, 0.17], color = "red")
plt.plot([0.51, 0.12, -0.06, 0.12, 0.17, 0.03, 0.37, -0.048, 0.06, 0.40], color = "blue")
plt.plot([0.29, -0.34, -0.26, 0.13, 0.26, -0.25, 0.10, 0.26, 0.44, -0.08], color = "green")
plt.plot([-0.46, 0.419, 0.75, 0.56, -0.33, 0.24, 0.12, 0.55, 0.29, 0.56], color = "orange")
plt.plot([0.25, -0.016, 0.14, 0.01, 0.52, 0.27, -0.20, 0.33, 0.27, 0.46], color = "yellow")
plt.plot([0.56, 0.021, 0.08, 0.72, 0.24, 0.32, 0.37, 0.36, 0.53, 0.38], color = "pink")

magetometers = [[0.40, 0.06, 0.28, -0.01, 0.39, 0.15, 0.01, 0.21, 0.63, 0.17],
[-0.46, 0.419, 0.75, 0.56, -0.33, 0.24, 0.12, 0.55, 0.29, 0.56]]

gradiometers = [[0.51, 0.12, -0.06, 0.12, 0.17, 0.03, 0.37, -0.048, 0.06, 0.40],
[0.29, -0.34, -0.26, 0.13, 0.26, -0.25, 0.10, 0.26, 0.44, -0.08],
[0.25, -0.016, 0.14, 0.01, 0.52, 0.27, -0.20, 0.33, 0.27, 0.46],
[0.56, 0.021, 0.08, 0.72, 0.24, 0.32, 0.37, 0.36, 0.53, 0.38]]

def calcAvg(Darray):
    final = []
    for i in range(len(Darray[0])):
        allVals = []
        for j in range(len(Darray)):
            allVals.append(Darray[j][i])
        final.append(sum(allVals)/len(allVals))
    return final
        
m1 = calcAvg(magetometers)
m2 = calcAvg(gradiometers)

plt.plot(m1, color = "red")
plt.plot(m2, color = "blue")


            

    
#plotCorrelation(200, 50, 70)
    
def analyseCorrelForIndicators(): 
    for i in range(15):
        plotCorrelation(i, 50, 70)

analyseCorrelForIndicators()
