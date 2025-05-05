import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from datetime import datetime
import kagglehub
import os

# Load and preprocess data
start_date = datetime(1999, 10, 20)
end_date = datetime(2040, 6, 20)

dataset_path = kagglehub.dataset_download("mczielinski/bitcoin-historical-data")
df = pd.read_csv(os.path.join(dataset_path, "btcusd_1-min_data.csv"))

df['Datetime'] = pd.to_datetime(df['Timestamp'], unit='s')
df.set_index('Datetime', inplace=True)
df = df.loc[start_date:end_date]

df = df['Close'].resample('D').median().to_frame()
df.dropna(inplace=True)

# Split data into train and test
split_index = int(len(df) * 0.8)
df_train = df.iloc[:split_index].copy()
df_test = df.iloc[split_index:].copy()

# Run auto_arima to find best order
model_auto = auto_arima(df_train, stepwise=True, seasonal=False,
                        suppress_warnings=True, trace=True,
                        information_criterion = "aic")
print("best ARIMA order found:", model_auto.order)

# Fit ARIMA with best order
model = ARIMA(df_train, order=model_auto.order, trend="t")
modelFit = model.fit()


residuals = modelFit.resid[1:]
fig, ax = plt.subplots(1,2)
residuals.plot(title = "Residuals", ax = ax[0])
ax[0].set_xlabel("Datetime")
ax[0].set_ylabel("Residuals")

residuals.plot(title = "Density", kind = "kde", ax = ax[1])
ax[1].set_xlabel("Residual Value")
ax[1].set_ylabel("Density")

plt.show()


# Forecast
forecastTest = modelFit.forecast(steps=len(df_test))

# Add forecast to dataframe for plotting
df["Predicted Price"] = np.nan
df.loc[df_test.index, "Predicted Price"] = forecastTest.values

# Plot
plt.plot(df['Close'], label='Actual Price', color='blue')  # Real price in blue
plt.plot(df['Predicted Price'], label='Predicted Price', color='red')  # Predicted in red
plt.axvline(df_test.index[0], label='Prediction Start', linestyle='dashed', color='black')
plt.title("ARIMA Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel("BTC Price (USD)")
plt.legend()
plt.tight_layout()
plt.show()


from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

mae = mean_absolute_error(df_test, forecastTest)
mape = mean_absolute_percentage_error(df_test, forecastTest)
mse = (mean_squared_error(df_test, forecastTest))

print(f'MAE: {mae}')
print(f'MAPE: {mape}')
print(f'MSE: {mse}')


# Save predictions to CSV
output_df = pd.DataFrame({
    'Datetime': df_test.index,
    'Actual': df_test['Close'].values,
    'Predicted': forecastTest.values
})

output_df.to_csv("ARIMAPredictedPrices.csv", index=False)
print("Predictions saved to ARIMAPredictedPrices.csv")