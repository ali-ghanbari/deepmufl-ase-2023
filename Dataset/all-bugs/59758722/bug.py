from keras import Sequential
from keras.layers import Dense
from sklearn.datasets import make_regression

X, y = make_regression(1000, 4)

model = Sequential()
model.add(Dense(30, activation='relu', input_shape=(4,)))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='relu'))

model.summary()

print('Compile & fit')
model.compile(loss='mean_squared_error', optimizer='RMSprop')
model.fit(X, y, batch_size=128, epochs=13)
