from keras import Sequential
from keras.layers.core import Activation
from keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.datasets import make_classification
import numpy as np

X, y = make_classification(1000, 900)
X = np.reshape(X, (-1, 30, 30, 1))

model = Sequential()
model.add(Conv2D(96, kernel_size=11, padding="same", input_shape=(30, 30, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Conv2D(128, kernel_size=3, padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(128, kernel_size=3, padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(units=300, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation("softmax"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=5)
