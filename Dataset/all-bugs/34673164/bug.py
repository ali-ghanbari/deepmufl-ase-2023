from keras.utils import to_categorical
from sklearn.datasets import make_classification
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

X_train, y_train = make_classification(1000, 14)
y_train = to_categorical(y_train)

model = Sequential()
model.add(Dense(64, input_dim=14, kernel_initializer='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, kernel_initializer='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(2, kernel_initializer='uniform'))
model.add(Activation('softmax'))

sgd = SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
model.fit(X_train, y_train, epochs=20, batch_size=16)
