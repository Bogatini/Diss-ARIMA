import csv
import datetime as datetime
import time
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from sklearn.metrics import mean_squared_error

# below recyled from Coin Price Predictor - this just gets the median close price per day between the two dates given

                             # y    m  d  h   m   s
startDate = datetime.datetime(1999, 1, 1, 00, 00, 00) # inclusive
#endDate   = datetime.datetime(2021, 1, 1, 00, 00, 00) # exclusive
endDate = datetime.datetime.now()

startDateUnix = time.mktime(startDate.timetuple()) # turn them into unix
endDateUnix = time.mktime(endDate.timetuple())

# download latest version of the data set
path = kagglehub.dataset_download("mczielinski/bitcoin-historical-data")

print("Path to dataset files:", path)

csvData = pd.read_csv(path + "\\btcusd_1-min_data.csv", header = 0)

csvData["Datetime"] = pd.to_datetime(csvData["Timestamp"], unit = "s")

timeSlice = csvData[(csvData["Timestamp"] >= startDateUnix) & (csvData["Timestamp"] < endDateUnix)]

timeSlice = timeSlice.copy()
timeSlice.loc[:, "Date"] = timeSlice["Datetime"].dt.date

dataGroup = timeSlice.groupby("Date")
closePriceGroup = dataGroup["Close"]

closePriceGroup = closePriceGroup.median()

closePriceGroup.index = pd.to_datetime(closePriceGroup.index)
closePriceGroup = closePriceGroup.asfreq("D")  # Set daily frequency


# Split into training and testing sets
trainSize = int(len(closePriceGroup) * 0.8)
trainData = closePriceGroup.iloc[:trainSize].values  # Convert to numpy array
testData = closePriceGroup.iloc[trainSize:].values


from statsmodels.tsa.stattools import adfuller
def adf_test(dataset):
    dftest = adfuller(dataset, autolag = "AIC")
    print("ADF : ", dftest[0])
    print("p-value : ", dftest[1])
    print("# lags : ", dftest[2])
    print("# observations for ADF regression:", dftest[3])
    print("critical values :")
    for key, val in dftest[4].items():
        print("\t",key, ": ", val)

adf_test(closePriceGroup)

from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

stepwise_fit = auto_arima(closePriceGroup, trace=True, suppress_warnings=True)
bestOrder = stepwise_fit.order

print(f"Best order found: {bestOrder}")

model = ARIMA(closePriceGroup, order=bestOrder)
model = model.fit()
#print(model.summary())                                                         ####

# Predictions on test data
start = len(trainData)
end = len(trainData) + len(testData) - 1
predictions = model.predict(start=start, end=end, typ="levels")

# Forecasting next 3 days
futureDays = 3
futurePredictions = model.forecast(steps=futureDays)

# Creating future dates
lastDate = closePriceGroup.index[-1]
futureDates = [lastDate + datetime.timedelta(days=i) for i in range(1, futureDays + 1)]

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(closePriceGroup.index[trainSize:], testData, label="Actual Prices", color="blue")
plt.plot(closePriceGroup.index[trainSize:], predictions, label="Past ARIMA Predictions", color="orange")
plt.plot(futureDates, futurePredictions, label="Future Forecast", color="red")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Median Close Price")
plt.title("Bitcoin Price Prediction using ARIMA")
plt.show()

for date, price in zip(futureDates, futurePredictions):
    print(f"Predicted Close Price on {date.date()}: {price}")

from sklearn.metrics import mean_squared_error
from math import sqrt

mse = mean_squared_error(predictions, testData)
rmse = mse**0.5
print(f"set mean: {closePriceGroup.mean()}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

#plt.plot(closePriceGroup.index[:trainSize], trainData, label="Training Data", color="blue")
#plt.plot(closePriceGroup.index[trainSize:], testData, label="Testing Data", color="orange")
#plt.show()