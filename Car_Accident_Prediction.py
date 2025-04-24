import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from pmdarima import auto_arima
from datetime import timedelta

plt.rcParams['figure.figsize'] = [10, 7.5]

path = kagglehub.dataset_download("sobhanmoosavi/us-accidents")
print("Path to dataset files:", path)

csv_path = f"{path}/US_Accidents_March23.csv"
accidents = pd.read_csv(csv_path)

# Display first few rows,columns
print(accidents.head())
print(accidents.columns)
accidents.info()

# # of miss values
missing_values_count = accidents.isnull().sum()
print("Missing Values Per Column:\n", missing_values_count)
total_cells = np.product(accidents.shape)
total_missing = missing_values_count.sum()
percent_missing = (total_missing / total_cells) * 100
print(f"Percentage of missing data: {percent_missing:.2f}%")

accidents["Start_Time"] = accidents["Start_Time"].astype(str)
accidents["End_Time"] = accidents["End_Time"].astype(str)

accidents["Start_Time_parsed"] = pd.to_datetime(accidents["Start_Time"], errors='coerce')
accidents["End_Time_parsed"] = pd.to_datetime(accidents["End_Time"], errors='coerce')

invalid_start_times = accidents[accidents["Start_Time_parsed"].isna()]
invalid_end_times = accidents[accidents["End_Time_parsed"].isna()]
accidents = accidents.dropna(subset=["Start_Time_parsed", "End_Time_parsed"])


# Check again for valid datetime conversion
print(accidents[["Start_Time_parsed", "End_Time_parsed"]].head())

# Extract date-related features
accidents["Hour"] = accidents["Start_Time_parsed"].dt.hour 
accidents["Day"] = accidents["Start_Time_parsed"].dt.weekday + 1  # Day of week (1=Monday, 7=Sunday)
accidents["Month"] = accidents["Start_Time_parsed"].dt.month  
accidents["Year"] = accidents["Start_Time_parsed"].dt.year  
accidents["Duration"] = (accidents["End_Time_parsed"] - accidents["Start_Time_parsed"]).dt.total_seconds() / 60 

# Print newly added columns
print(accidents[["Hour", "Day", "Month", "Year", "Duration"]].describe())

# Drop unnecessary columns
columns_to_drop = [
    'ID', 'Source', 'Description', 'Street', 'Zipcode', 'Country',
    'Timezone', 'Airport_Code', 'End_Time',
    'Start_Lat', 'Start_Lng', 'End_Lat', 'End_Lng', 'Distance(mi)',
    'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit',
    'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Turning_Loop',
    'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight',
    'Weather_Timestamp']

accidents.drop(columns=columns_to_drop, inplace=True, errors='ignore')

print("Dataset after dropping columns:")
print(accidents.info())


# Create daily time series
accidents['Date'] = accidents['Start_Time_parsed'].dt.date
time_series = accidents.groupby('Date').size()
time_series.index = pd.to_datetime(time_series.index)
print("\nTime series head:")
print(time_series.head())

# Plot the time series
plt.figure(figsize=(14, 6))
plt.plot(time_series, label='Daily Accidents')
plt.title("Daily Traffic Accidents Over Time", fontsize=16)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Accident Count", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('daily_accidents_timeseries.png')
plt.show()

# Plot monthly accidents
monthly_accidents = accidents.groupby(pd.Grouper(key='Start_Time_parsed', freq='M')).size()
plt.figure(figsize=(14, 6))
plt.plot(monthly_accidents.index, monthly_accidents.values, marker='o', linestyle='-')
plt.title("Monthly Traffic Accidents", fontsize=16)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Accident Count", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('monthly_accidents.png')
plt.show()

# Plot weekly accidents
daily_avg_by_weekday = accidents.groupby('Day').size() / len(accidents['Date'].unique())
plt.figure(figsize=(10, 6))
sns.barplot(x=daily_avg_by_weekday.index, y=daily_avg_by_weekday.values)
plt.title("Average Daily Accidents by Day of Week", fontsize=16)
plt.xlabel("Day of Week (1=Monday, 7=Sunday)", fontsize=14)
plt.ylabel("Average Accidents", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('accidents_by_weekday.png')
plt.show()

# Plot ACF and PACF
plt.figure(figsize=(16, 6))
plt.subplot(121)
plot_acf(time_series.dropna(), lags=50, ax=plt.gca())
plt.title("Autocorrelation Function (ACF)", fontsize=14)
plt.subplot(122)
plot_pacf(time_series.dropna(), lags=50, ax=plt.gca())
plt.title("Partial Autocorrelation Function (PACF)", fontsize=14)
plt.tight_layout()
plt.savefig('acf_pacf_plots.png')
plt.show()


# model evaluation
#RMSE Root Mean Squared Error (less sensitive to outliers)
#Mean Absolute Percentage Error
#MAPE is how much prediction deviate from actual value
def evaluate_forecast(actual, predicted, model_name):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    # Add SMAPE calculation
    smape = 100 * np.mean(
        2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted))
    )

    print(f"\n{model_name} Model Evaluation:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"SMAPE: {smape:.2f}%")
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape, 'smape': smape}


# Step 1: Aggregate to monthly accident counts for ARIMA
monthly_series = accidents.groupby(pd.Grouper(key='Start_Time_parsed', freq='M')).size()
monthly_series.index = pd.to_datetime(monthly_series.index)
# Step 2: ADF test for stationarity
adf_result = adfuller(monthly_series)
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
if adf_result[1] > 0.05:
    print("=> Non-stationary. Differencing required.")
else:
    print("=> Series is stationary.")

monthly_series_diff = monthly_series.diff().dropna()
monthly_series_seasonal_diff = monthly_series.diff(12).dropna()  # 12 for monthly data


# Auto ARIMA for best (p,d,q) [tried didnt perform well]
auto_model = pm.auto_arima(monthly_series,
                           seasonal=True,
                           stepwise=True,
                           suppress_warnings=True,
                           trace=True)

print("Best ARIMA order:", auto_model.order)

#ARIMA model, p = autoregressive, q = lags
p, d, q = auto_model.order
model = ARIMA(monthly_series, order=(40, 1, 6))
model_fit = model.fit()
print(model_fit.summary())

# Forecasting and plotting
n_months = 12
forecast = model_fit.get_forecast(steps=n_months)
forecast_values = forecast.predicted_mean
conf_int = forecast.conf_int()
plt.figure(figsize=(12, 6))
plt.plot(monthly_series.index, monthly_series, label='Historical')
plt.plot(forecast_values.index, forecast_values, label='Forecast', color='green')
plt.fill_between(forecast_values.index,
                 conf_int.iloc[:, 0],
                 conf_int.iloc[:, 1],
                 color='lightgreen',
                 alpha=0.3)
plt.title("ARIMA Forecast of Monthly Accidents (Next 12 Months)")
plt.xlabel("Month")
plt.ylabel("Predicted Accident Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#evaluate the series
if len(monthly_series) >= n_months * 2:
    actual = monthly_series[-n_months:]
    predicted = model_fit.predict(start=len(monthly_series) - n_months, end=len(monthly_series) - 1)
    metrics = evaluate_forecast(actual, predicted, "ARIMA")



