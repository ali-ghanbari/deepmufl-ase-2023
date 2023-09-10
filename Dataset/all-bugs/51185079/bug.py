from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_classification

x, y = make_classification(1000, 34)

model = Sequential()
model.add(Dense(512, activation='relu', input_dim=34))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x, y, epochs=2, batch_size=200)
