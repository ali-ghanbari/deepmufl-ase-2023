from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

dataset = loadtxt("dataset.csv", delimiter=",")
X = dataset[:, 0:2]
y = dataset[:, 2]
model = Sequential()
model.add(Dense(196, input_dim=2, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=600, batch_size=10)
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy * 100))
