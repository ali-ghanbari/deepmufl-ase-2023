from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_regression

X, y = make_regression(100, 18, random_state=42)

model = Sequential()
model.add(Dense(18, input_dim=18, activation='tanh'))
model.add(Dense(36, activation='relu'))
model.add(Dense(72, activation='relu'))
model.add(Dense(72, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='softmax'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model
model.fit(X, y, epochs=100, batch_size=35)
