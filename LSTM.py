import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Load+preprocess
accidents = pd.read_csv("US_Accidents_March23.csv")
accidents['Start_Time_parsed'] = pd.to_datetime(accidents['Start_Time'], errors='coerce')
accidents['End_Time_parsed']   = pd.to_datetime(accidents['End_Time'],   errors='coerce')
accidents = accidents.dropna(subset=['Start_Time_parsed', 'End_Time_parsed'])

# Extract duration and time features
accidents['Date']     = accidents['Start_Time_parsed'].dt.date
accidents['Hour']     = accidents['Start_Time_parsed'].dt.hour
accidents['Duration'] = (
    (accidents['End_Time_parsed'] - accidents['Start_Time_parsed'])
    .dt.total_seconds() / 60
)
#group useful features
agg_numeric = accidents.groupby('Date').agg({
    'Temperature(F)':     'mean',
    'Humidity(%)':        'mean',
    'Pressure(in)':       'mean',
    'Visibility(mi)':     'mean',
    'Wind_Speed(mph)':    'mean',
    'Precipitation(in)':  'sum',
    'Duration':           'mean',
    'Hour':               'median',
    'Severity':           'mean'
})

daily_counts = accidents.groupby('Date').size().rename('Accident_Count')
weather_mode = accidents.groupby('Date')['Weather_Condition'] \
    .agg(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown')
daily_df = pd.concat([agg_numeric, daily_counts, weather_mode.rename('Weather_Condition')], axis=1)
daily_df.index = pd.to_datetime(daily_df.index)
daily_df = daily_df.sort_index()
daily_df['Day_of_Week'] = daily_df.index.dayofweek
daily_df['Month']       = daily_df.index.month
daily_df['Is_Weekend']  = daily_df['Day_of_Week'].isin([5,6]).astype(int)
daily_df = pd.get_dummies(daily_df, columns=['Weather_Condition'], drop_first=True)
features = daily_df.drop(columns=['Accident_Count'])
target   = daily_df['Accident_Count']
#Normalization Process
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(features)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(target.values.reshape(-1,1))
# Define sequence for 30 days in respect to a month
# number of times the stacked LSTMs run among themselves creating the sequence
SEQ_LEN = 365
#sequences of data for time series modelin

def create_sequences(X, y, seq_len=SEQ_LEN):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)
#inputs
X_seq, y_seq = create_sequences(X_scaled, y_scaled)
#train/testing split
X_train, y_train = X_seq[:-365], y_seq[:-365]
X_test,  y_test  = X_seq[-365:], y_seq[-365:]
model = Sequential([
    LSTM(128, input_shape=(SEQ_LEN, X_train.shape[2])),
    Dropout(0.2),
    Dense(1)
])
#MSE loss
model.compile(optimizer='adam', loss='mse')
#training and parameter adjusting
model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
y_true = scaler_y.inverse_transform(y_test).flatten()
test_dates = daily_df.index[-len(y_true):]
#same thing as evaluation but for troubleshooting cuz I was running to problems
rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
mae   = mean_absolute_error(y_true, y_pred)
mask = y_true >= 10
#1/n sum[(true - prediction)/true]*100
mape_filtered = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
#100/n sum[(prediction - true)/((true+pred)/2)]
smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
print(f"Test RMSE         : {rmse:.2f}")
print(f"Test MAE          : {mae:.2f}")
print(f"Filtered MAPE     : {mape_filtered:.2f}%")
print(f"Symmetric MAPE    : {smape:.2f}%")

forecast_dates = pd.date_range(start="2024-01-01", periods=365, freq='D')
feature_cols = features.columns.tolist()
#fill averages 
future = pd.DataFrame(index=forecast_dates, columns=feature_cols, dtype=float)
future['Day_of_Week'] = future.index.dayofweek
future['Month']       = future.index.month
future['Is_Weekend']  = future['Day_of_Week'].isin([5,6]).astype(int)

for col in ['Temperature(F)','Humidity(%)','Pressure(in)',
            'Visibility(mi)','Wind_Speed(mph)','Precipitation(in)',
            'Duration','Hour','Severity']:
    future[col] = daily_df[col].mean()
for col in feature_cols:
    if col.startswith('Weather_Condition_'):
        future[col] = 0

future_scaled = scaler_X.transform(future)
last_seq = X_seq[-1].copy()
forecast_scaled = []
#predict 365 days, uses previous input values to forecast
for feat_row in future_scaled:
    p = model.predict(last_seq[np.newaxis, :, :])[0,0]
    forecast_scaled.append(p)
    last_seq = np.vstack([last_seq[1:], feat_row])
#inverse is needed due to the scaler vectors
forecast = scaler_y.inverse_transform(
    np.array(forecast_scaled).reshape(-1,1)
).flatten()
#plot LSTM
plt.figure(figsize=(12,6))
plt.plot(daily_df.index, daily_df['Accident_Count'], label='Historical', alpha=0.5)
plt.plot(test_dates, y_true, label='Actual (Test)', color='black')
plt.plot(test_dates, y_pred, label='Predicted (Test)', linestyle='--')
plt.plot(forecast_dates, forecast, label='LSTM Forecast (2025)', color='green')
plt.title("LSTM: Actual vs Predicted vs 2025 Forecast")
plt.xlabel("Date")
plt.ylabel("Daily Accident Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
