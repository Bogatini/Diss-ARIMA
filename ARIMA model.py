import csv
import datetime as datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from itertools import product

# NOTE!! get rid of warnings theyre annoying but remove the removal later

input_csv = "btcusd_1-min_data.csv" # dont do this - fetch it from online (kagglehub) - check AI getter

                             # y    m  d  h   m   s
startDate = datetime.datetime(2023, 1, 1, 00, 00, 00) # inclusive
endDate   = datetime.datetime(2024, 1, 1, 00, 00, 00) # exclusive

startDate = time.mktime(startDate.timetuple()) # turn them into unix
endDate = time.mktime(endDate.timetuple())

csvData = pd.read_csv(input_csv, header = 0)

csvData["Datetime"] = pd.to_datetime(csvData["Timestamp"], unit = "s")

timeSlice = csvData[(csvData["Timestamp"] >= startDate) & (csvData["Timestamp"] < endDate)]

trainingDataSlice = timeSlice[:int(len(timeSlice)*0.8)]
testDataSlice = timeSlice[int(len(timeSlice)*0.8):]

# define ARIMA parameters - we will need to find the most efficient parameters to pass in
# the higher these numbers the longer trainging will take as we check every possible set of parameters
p = range(0,4)      # autoregressive order - number of previous values used to predict the current value - find using partial autocorrelation function (PACF)
d = range(0,2)      # differencing order - number of times the data is "differenced" (removing trends / seasonality) - this can be explored more in SARIMA
q = range(0,4)      # Moving average order - number of past foecast errors included in the model. Found using autocorrelation function (ACF)

def evaluateARIMAModel(trainSet, testSet, ARIMAOrder):
    model = ARIMA(trainSet, order = ARIMAOrder)
    fittedModel = model.fit()
    modelPredictions = fittedModel.forecast(steps=len(testSet))
    meanSquaredError = mean_squared_error(modelPredictions, testSet)

    return meanSquaredError, fittedModel

#  https://www.youtube.com/watch?v=aZmYr71YiWQ

bestMSE = 999999999999999999  # !!!*** make this inf wtf?

results = []
for p, d, q in product(p,d,q):
    ARIMAOrder = (p,d,q)

    meanSquaredError, fittedModel = evaluateARIMAModel(trainingDataSlice["Close"], testDataSlice["Close"], ARIMAOrder)

    results.append((ARIMAOrder, meanSquaredError, fittedModel))



    if fittedModel: # there are many reasons why a set of parameters will just return None
        #plt.figure()
        #plt.plot(timeSlice["Datetime"][:len(trainingDataSlice)], trainingDataSlice["Close"])
        #plt.plot(timeSlice["Datetime"][int(len(timeSlice)*0.8):], testDataSlice["Close"])
        #plt.plot(timeSlice["Datetime"][int(len(timeSlice)*0.8):], fittedModel.forecast(steps = len(testDataSlice)), label=f"{ARIMAOrder}")
        # title
        # labels
        #plt.legend()
        #plt.show()


        if meanSquaredError < bestMSE:
            bestMSE = meanSquaredError

print(bestMSE)

# 109445837.08396368
# 109445837

#plt.plot(timeSlice["Datetime"][:len(trainingDataSlice)], trainingDataSlice["Close"])
#plt.plot(timeSlice["Datetime"][int(len(timeSlice)*0.8):], testDataSlice["Close"])
#plt.show()
