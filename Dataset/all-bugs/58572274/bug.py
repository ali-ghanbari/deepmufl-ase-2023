from keras.models import Sequential
from keras.layers import Dense
import numpy

data = numpy.array([[0., 0., 1.], [0., 1., 0.], [1., 0., 0.], [1., 1., 1.]])
train = data[:, :-1]  # Taking The same and All data for training
test = data[:, :-1]

train_l = data[:, -1]
test_l = data[:, -1]

train_label = []
test_label = []

for i in train_l:
    train_label.append([i])
for i in test_l:
    test_label.append([i])  # Just made Labels Single element...

train_label = numpy.array(train_label)
test_label = numpy.array(test_label)  # Numpy Conversion

model = Sequential()

model.add(Dense(2, input_dim=2, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer='adam')

model.fit(train, train_label, epochs=10, verbose=2)

model.predict_classes(test)
