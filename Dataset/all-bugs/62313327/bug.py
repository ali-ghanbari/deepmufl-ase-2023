import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
import tensorflow as tf
np.random.seed(123)
dataSize = 100000
xdata = np.zeros((dataSize, 30))
ydata = np.zeros(dataSize)
for i in range(dataSize):
    vec = (np.random.rand(30) * 2) - 1
    xdata[i] = vec
    ydata[i] = vec[1]
model = models.Sequential()
model.add(layers.Dense(1, activation="relu", input_shape=(30, )))
model.add(layers.Dense(1, activation="sigmoid"))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
lossObject = tf.keras.losses.MeanSquaredError()
model.compile(optimizer=optimizer, loss=lossObject)
model.fit(xdata, ydata, epochs=200, batch_size=32)
