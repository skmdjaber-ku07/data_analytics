import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('dataset.csv')

data['Date'] = pd.to_datetime(data['Date'])

data.sort_values(by='Date', inplace=True)

data.set_index('Date', inplace=True)
close_values = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled_close_values = scaler.fit_transform(close_values)
window_size = 30
X = []
y = []
for i in range(window_size, len(scaled_close_values)):
    X.append(scaled_close_values[i-window_size:i, 0])
    y.append(scaled_close_values[i, 0])
X = np.array(X)
y = np.array(y)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape the input data to be 3-dimensional for LSTM input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, epochs=100, batch_size=32)

test_loss = model.evaluate(X_test, y_test)
print('Test loss:', test_loss)

new_data = [100, 102, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240]
new_data = np.array(new_data).reshape(-1, 1)
scaled_new_data = scaler.transform(new_data)
input_sequence = scaled_new_data[-window_size:].reshape(1, -1, 1)
predicted_output = model.predict(input_sequence)
predicted_output = scaler.inverse_transform(predicted_output)
print('Predicted output:', predicted_output[0][0])
