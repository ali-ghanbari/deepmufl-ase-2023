from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.datasets import make_classification

X, y = make_classification(1000, 283, n_classes=4, n_informative=4)
y = to_categorical(y)

model = Sequential()
model.add(Dense(100, input_dim=283, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(150, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, kernel_initializer='normal', activation='sigmoid'))
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X, y, epochs=5)
