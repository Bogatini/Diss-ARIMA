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

# Split into training and testing sets
trainSize = int(len(closePriceGroup) * 0.8)
trainData = closePriceGroup.iloc[:trainSize].values  # Convert to numpy array
testData = closePriceGroup.iloc[trainSize:].values

# https://people.duke.edu/~rnau/411arim.htm

# ARIMA is split into three sections:
# AR - autoregression: Lags of the stationarized series in the forecasting equation are called "autoregressive" terms
# I - integrated: a time series which needs to be differenced to be made stationary is said to be an "integrated" version of a stationary series
# MA - moving average: lags of the forecast errors are called "moving average" terms


# Define ARIMA parameters
p_range = range(0, 4)  # the number of autoregressive terms
d_range = range(0, 2)  # the number of nonseasonal differences needed for stationarity
q_range = range(0, 4)  # the number of lagged forecast errors in the prediction equation.

def difference_series(series, order=1):
    # apply differencing to make the series stationary.
    return np.diff(series, n=order)

def inverse_difference(original, diff_series):
    # revert differenced series back to original scale
    return np.cumsum(np.insert(diff_series, 0, original[0]))

def fit_ar_model(data, p):
    # fit an Autoregressive (AR) model of order p using least squares
    X = np.array([data[i-p:i] for i in range(p, len(data))])
    y = data[p:]
    if len(X) == 0 or len(y) == 0:
        return np.zeros(p)  # If not enough data, return zeros
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]  # Solve Ax = b
    return coeffs

def forecast_ar(coeffs, history, steps):
    """Forecast using AR model."""
    predictions = []
    history = list(history)
    for _ in range(steps):
        pred = np.dot(coeffs, history[-len(coeffs):])
        predictions.append(pred)
        history.append(pred)
    return np.array(predictions)

def moving_average(errors, q):
    # Fit a simple Moving Average (MA) model.
    if len(errors) < q:
        return np.zeros(q)
    coeffs = np.linalg.lstsq(
        np.array([errors[i-q:i] for i in range(q, len(errors))]),
        errors[q:],
        rcond=None
    )[0]
    return coeffs

def forecast_ma(errors, ma_coeffs, steps):
    # forecast future errors using MA model
    predicted_errors = []
    history = list(errors)
    for _ in range(steps):
        pred_error = np.dot(ma_coeffs, history[-len(ma_coeffs):]) if len(ma_coeffs) > 0 else 0
        predicted_errors.append(pred_error)
        history.append(pred_error)
    return np.array(predicted_errors)

def evaluate_arima(train, test, p, d, q):
    # fit and evaluate the ARIMA (p,d,q) model.
    if d > 0:
        train_diff = difference_series(train, d)
    else:
        train_diff = train.copy()

    # Fit AR and MA models
    ar_coeffs = fit_ar_model(train_diff, p)
    errors = train_diff[p:] - forecast_ar(ar_coeffs, train_diff[:p], len(train_diff) - p)
    ma_coeffs = moving_average(errors, q)

    # Forecasting
    test_diff = difference_series(test, d) if d > 0 else test.copy()
    ar_forecast = forecast_ar(ar_coeffs, train_diff, len(test_diff))
    ma_forecast = forecast_ma(errors, ma_coeffs, len(test_diff))
    predictions = ar_forecast + ma_forecast

    if d > 0:
        predictions = inverse_difference(train[-len(predictions)-1:], predictions)

    mse = mean_squared_error(test, predictions)
    return mse, predictions

# Iterate over parameter grid
bestMSE = float("inf")
bestOrder = None
bestPredictions = None

for p, d, q in product(p_range, d_range, q_range):
    try:
        mse, predictions = evaluate_arima(trainData, testData, p, d, q)
        if mse < bestMSE:
            bestMSE = mse
            bestOrder = (p, d, q)
            bestPredictions = predictions
    except Exception as e:
        print(f"ARIMA({p},{d},{q}) failed: {e}")

# Print best model
print(f"Best ARIMA Order: {bestOrder}, MSE: {bestMSE}")

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(closePriceGroup.index[:trainSize], trainData, label="Training Data")
plt.plot(closePriceGroup.index[trainSize:], testData, label="Testing Data", color="orange")

if bestPredictions is not None:
    plt.plot(closePriceGroup.index[trainSize:], bestPredictions, label=f"ARIMA{bestOrder}", linestyle="dashed")

plt.title("Bitcoin Daily Close Price Forecast")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()
