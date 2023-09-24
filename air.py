import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_predict

from datetime import datetime

data = pd.read_csv('./AirPassengers.csv')
data['Month'] = pd.to_datetime(data["Month"], infer_datetime_format=True)

# print(data.head())
# print(data.info())
# exit()
#
# plt.xlabel("Data")
# plt.ylabel("No of Passengers")
# plt.plot(data['#Passengers'])
# plt.show()
# exit()

def check_stationary(ts_data):
    df_test = adfuller(ts_data)
    output = pd.Series(df_test[0:4], index=["Test statistic", "p-value", "used_lag", "NOBS"])

    print(output)


# 一阶差分 0.054, 二阶差分 0.038, d=2(判断值0.05)
df1 = data['#Passengers'].diff(1).dropna()
check_stationary(df1)
df2 = df1.diff(1).dropna()
check_stationary(df2)

# plt.xlabel("Data")
# plt.ylabel("No of Passengers")
# plt.plot(df1)
# plt.plot(df2)
# plt.plot(df3)
# _, ax = plt.subplots(1, 2, 3)
# df1.plot(title="first order", ax=ax[0])
# residuals.plot(kind='kde', title='Density', ax=ax[1])
# plt.show()
d = 2

# pacf 计算p
# fig, ax = plt.subplots(figsize=(10, 6))
# plot_acf(df2.dropna(), ax=ax)
q = 2
# plot_pacf(df2.dropna(), ax=ax)
# plt.show()

p = 11

# ARIMA(12,2,2)

model = ARIMA(data['#Passengers'].values, order=(p, d, q))
# estimate the parameters of the model
model_fit = model.fit()
# print(model_fit.summary())

# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
_, ax = plt.subplots(1, 2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

# prediction
start_date = data['Month'].index[0]
end_date = data['Month'].index[-1]

predict = model_fit.predict(start=start_date, end=end_date)
data['Predicted_ARIMA'] = predict

# data[['#Passengers', 'Predicted_ARIMA']].plot()

# Plot actual vs. fitted values
# plt.figure(figsize=(10, 6))
# plot_predict(model_fit, dynamic=False)
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.title('Actual vs. Fitted Values')
# plt.legend(loc='upper left')
# plt.show()


# Define the number of steps to forecast
forecast_steps = 10

# Forecast future values
forecast = model_fit.forecast(steps=forecast_steps)

# Create a range of future dates
date_range = pd.date_range(start=data['Month'].iloc[-1], periods=forecast_steps + 1, freq='MS')[1:]

# Create a DataFrame for the forecasted values
forecast_df = pd.DataFrame({'Month': date_range, 'forecast': forecast})

# Merge the forecasted values with the original data
data = pd.concat([data, forecast_df], ignore_index=True)

data[['#Passengers', 'forecast']].plot()
plt.show()
