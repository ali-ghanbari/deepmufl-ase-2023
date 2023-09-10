import numpy as np
from keras.layers import Dense
from keras.callbacks import History
from keras import optimizers, Sequential

X = np.array([[1], [2]])
Y = np.array([[1], [0]])

history = History()

inputDim = len(X[0])
print('input dim', inputDim)
model = Sequential()

model.add(Dense(1, activation='sigmoid', input_dim=inputDim))
model.add(Dense(1, activation='sigmoid'))

sgd = optimizers.SGD(lr=0.009, decay=1e-10, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X, Y, validation_split=0.1, verbose=2, callbacks=[history], epochs=20, batch_size=32)
