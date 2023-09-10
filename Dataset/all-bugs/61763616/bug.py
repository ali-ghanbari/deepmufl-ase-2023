from keras.models import Sequential
from keras.layers import MaxPooling1D, Conv1D
from keras.layers import Flatten, Dense
from sklearn.datasets import make_classification

X_train, y_train = make_classification(1000, 28 * 28)

model_05_01 = Sequential()
model_05_01.add(Conv1D(filters=16, kernel_size=12, input_shape=(X_train.shape[1], 1)))
model_05_01.add(MaxPooling1D(pool_size=4))

model_05_01.add(Conv1D(filters=32, kernel_size=12))
model_05_01.add(MaxPooling1D(pool_size=4))

model_05_01.add(Conv1D(filters=16, kernel_size=12))
model_05_01.add(MaxPooling1D(pool_size=4))

model_05_01.add(Flatten())

model_05_01.add(Dense(16, activation='relu'))
model_05_01.add(Dense(2, activation='sigmoid'))

model_05_01.compile(loss='logcosh', optimizer='adam', metrics=['accuracy'])

model_05_01.fit(X_train, y_train, epochs=5)
