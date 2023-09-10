import numpy as np
from keras import Sequential
from keras.layers import GRU, Dense

data = [[[0], [0], [0], [1], [1], [0], [1], [1]]]
output = [[[0], [0], [0], [1], [0], [1], [1], [0]]]

model = Sequential()
model.add(GRU(10, input_shape=(8, 1), return_sequences=True))
model.add(GRU(10, return_sequences=True))
model.add(Dense(1, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(np.asarray(data), np.asarray(output), epochs=3000)
