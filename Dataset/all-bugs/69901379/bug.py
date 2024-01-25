import numpy as np
import tensorflow as tf
from keras.optimizers import Adam

n = 50
x = np.random.randint(50, 2000, (n, 10))
y = np.random.randint(600, 4000, (n, 1))

k = 16

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(10,)),
    tf.keras.layers.Dense(k, activation='relu'),
    tf.keras.layers.Dense(k, activation='relu'),
    tf.keras.layers.Dense(k, activation='relu'),
    tf.keras.layers.Dense(k, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss='mse', optimizer=Adam(learning_rate=0.0001))
history = model.fit(x, y, epochs=1000, batch_size=32, verbose=0)

loss = history.history['loss']

for epoch in [1, 10, 50, 100, 500, 1000]:
    print('Epoch: {}, Loss: {:,.4f}'.format(epoch, loss[epoch - 1]))
