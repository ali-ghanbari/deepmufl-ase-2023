import keras.optimizers
from keras.models import Sequential
from keras.layers import Dense
import numpy

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# split into input (X) and output (Y) variables
X = numpy.array([[0.5, 1, 1], [0.9, 1, 2], [0.8, 0, 1], [0.3, 1, 1], [0.6, 1, 2], [0.4, 0, 1], [0.9, 1, 7], [0.5, 1, 4],
                 [0.1, 0, 1], [0.6, 1, 0], [1, 0, 0]])
y = numpy.array([[1], [1], [1], [2], [2], [2], [3], [3], [3], [0], [0]])

# create model
model = Sequential()
model.add(Dense(3, input_dim=3, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# Compile model
opt = keras.optimizers.SGD(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Fit the model
model.fit(X, y, epochs=150)
# calculate predictions
predictions = model.predict(X)
# round predictions
predictions_class = predictions.argmax(axis=-1)
print(predictions_class)
