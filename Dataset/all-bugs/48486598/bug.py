from keras import Sequential
from keras.layers import Dense
import numpy as np

# declare and init arrays for training-data

X = np.arange(0.0, 10.0, 0.05)
Y = np.empty(shape=0, dtype=float)

# Calculate Y-Values
for x in X:
    Y = np.append(Y, float(0.05*(15.72807*x - 7.273893*x**2 + 1.4912*x**3 - 0.1384615*x**4 + 0.00474359*x**5)))

# model architecture
model = Sequential()
model.add(Dense(1, input_shape=(1,)))
model.add(Dense(5))
model.add(Dense(1, activation='linear'))

# compile model
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

# train model
model.fit(X, Y, epochs=150, batch_size=10)
# declare and init arrays for prediction
YPredict = np.empty(shape=0, dtype=float)

# Predict Y
YPredict = model.predict(X)

print(YPredict)
