from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
import tensorflow.keras.activations as tfnn

from sklearn.datasets import load_breast_cancer

data, target = load_breast_cancer(return_X_y=True)

model = Sequential()

model.add(Dense(30, activation=tfnn.relu, input_dim=30))
model.add(BatchNormalization(axis=1))

model.add(Dense(60, activation=tfnn.relu))
model.add(BatchNormalization(axis=1))

model.add(Dense(1, activation=tfnn.softmax))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(data, target, epochs=6)
