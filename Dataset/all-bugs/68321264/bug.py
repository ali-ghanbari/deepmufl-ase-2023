import numpy as np
from keras import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X = []
y = []
for i in range(len(y_train)):
    if y_train[i] < 3:
        X.append(X_train[i])
        y.append(y_train[i])
for i in range(len(y_test)):
    if y_test[i] < 3:
        X.append(X_test[i])
        y.append(y_test[i])
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_train = np.expand_dims(X_train, axis=3)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=X_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3, 3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('softmax'))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=24, epochs=3, validation_split=0.1)

predictions = model.predict(X_test)
