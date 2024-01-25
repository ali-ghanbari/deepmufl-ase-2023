import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

data = [[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]
output = [[1, 0], [0, 1], [0, 1], [1,0]]

model = Sequential()
model.add(LSTM(10, input_shape=(1, 2), return_sequences=True))
model.add(LSTM(10))
model.add(Dense(2))

model.compile(loss='mae', optimizer='adam')
model.fit(np.asarray(data), np.asarray(output), epochs=50)

print(model.predict_classes(np.asarray(data)))
