from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import numpy

X = numpy.array([[1., 1.], [0., 0.], [1., 0.], [0., 1.], [1., 1.], [0., 0.]])
y = numpy.array([[0.], [0.], [1.], [1.], [0.], [0.]])
model = Sequential()
model.add(Dense(2, input_dim=2, kernel_initializer='uniform', activation='sigmoid'))
model.add(Dense(3, kernel_initializer='uniform', activation='sigmoid'))
model.add(Dense(1, kernel_initializer='uniform', activation='softmax'))
sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(X, y, epochs=20)
print()
score = model.evaluate(X, y)
print()
print(score)
print(model.predict(numpy.array([[1, 0]])))
print(model.predict(numpy.array([[0, 0]])))
