import keras
from sklearn.datasets import make_regression

X, y = make_regression(1000, 424 * 424, n_targets=3, n_informative=3, random_state=42)

model = keras.Sequential()
# explicitly define SGD so that I can change the decay rate
sgd = keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.add(keras.layers.Dense(32, input_shape=(424 * 424,)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dense(3, activation='softmax'))

model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)
