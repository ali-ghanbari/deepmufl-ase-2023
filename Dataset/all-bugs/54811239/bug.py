import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dense

X = [[[8, 0, 18, 10],
      [9, 0, 20, 7],
      [7, 0, 17, 12]],
     [[7, 0, 31, 8],
      [5, 0, 22, 9],
      [7, 0, 17, 12]]]
y = [[[10],
      [7],
      [12]],
     [[8],
      [9],
      [12]]]

X = np.asarray(X)
y = np.asarray(y)

model = Sequential()
model.add(LSTM(10, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dense(50, activation='tanh'))
model.add(Dense(1, activation='tanh'))

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit(X, y, epochs=5)
