from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

dataset = np.loadtxt("cars.csv", delimiter=",")
x = dataset[:, 0:5]
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
y = dataset[:, 5]
y = np.reshape(y, (-1, 1))

model = Sequential()
model.add(Dense(12, input_dim=5, kernel_initializer='normal', activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', 'accuracy'])
model.fit(x, y, epochs=150, batch_size=50, verbose=1, validation_split=0.2)
