from keras.layers import GRU, Dropout, Dense
from keras.models import Sequential
import numpy as np

data_len = 300
x = []
y = []
for i in range(data_len):
    a = np.random.randint(1,10,5)
    if a[0] % 2 == 0:
        y.append('0')
    else:
        y.append('1')

    a = a.reshape(5, 1)
    x.append(a)
    # print(x)

X = np.array(x)
Y = np.array(y)

model = Sequential()
model.add(GRU(units=24, activation='relu', return_sequences=True, input_shape=[5,1]))
model.add(Dropout(rate=0.5))
model.add(GRU(units=12, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=1, activation='softmax'))

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.summary()

history = model.fit(X[:210], Y[:210], epochs=20, validation_split=0.2)
