import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv1D, MaxPooling1D
from sklearn.datasets import make_classification

X, Y = make_classification(549, 600, random_state=42)

X1 = X[:90]
X = X[91:]
Y1 = Y[:90]
Y = Y[91:]
X = np.expand_dims(X, axis=2)
X1 =np.expand_dims(X1, axis=2)
print(np.array(X).shape)

model = Sequential()
model.add(Conv1D(filters=20, kernel_size=4, activation='relu', padding='same', input_shape=(600, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(50, activation='relu', input_dim=600))
model.add(Dense(1, activation='softmax'))

model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=['accuracy'])

model.fit(np.array(X), np.array(Y), epochs=100, batch_size=8, verbose=1, validation_data=(np.array(X1), np.array(Y1)))

scores = model.evaluate(np.array(X1), np.array(Y1), verbose=0)
