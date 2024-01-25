import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation

x_train = np.asarray([[.5], [1.0], [.4], [5], [25]])
y_train = np.asarray([.25, .5, .2, 2.5, 12.5])

opt = keras.optimizers.Adam(lr=0.01)

model = Sequential()
model.add(Dense(1, activation="relu", input_shape=(x_train.shape[1:])))
model.add(Dense(9, activation="relu"))
model.add(Dense(1, activation="relu"))

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])
model.fit(x_train, y_train, shuffle=True, epochs=10)

print(model.predict(np.asarray([[5]])))
