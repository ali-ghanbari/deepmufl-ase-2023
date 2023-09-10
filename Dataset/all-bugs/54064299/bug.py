from keras import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten
from sklearn.datasets import make_regression

X, y = make_regression(1000, 150 * 150 * 1, n_targets=4, random_state=42)
X = X.reshape(-1, 150, 150, 1)

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=(150, 150, 1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Dense(64))
model.add(Activation("relu"))

model.add(Flatten())
model.add(Dense(4))

model.compile(loss="mean_squared_error", optimizer='adam', metrics=[])

model.fit(X, y, batch_size=1, validation_split=0, epochs=30, verbose=1)
