import tensorflow as tf
import keras
from sklearn.datasets import make_regression

X, y = make_regression(100, 1323000)
X = X.reshape(-1, 1323000, 1)

model = keras.Sequential([
    keras.layers.Conv1D(filters=100, kernel_size=10000, strides=5000, input_shape=(1323000, 1), activation='relu'),
    keras.layers.Conv1D(filters=100, kernel_size=10, strides=3, input_shape=(263, 100), activation='relu'),
    keras.layers.LSTM(1000),
    keras.layers.Dense(500, activation='relu'),
    keras.layers.Dense(250, activation='relu'),
    keras.layers.Dense(1, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(X, y, epochs=1000)
