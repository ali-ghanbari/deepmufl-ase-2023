import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer

batch_size = 32
epochs = 10
alpha = 0.0001
lambda_ = 0
h1 = 50

train = pd.read_csv('mnist_train.csv.zip')
test = pd.read_csv('mnist_test.csv.zip')

train = train.loc['1':'5000', :]
test = test.loc['1':'2000', :]

train = train.sample(frac=1).reset_index(drop=True)
test = test.sample(frac=1).reset_index(drop=True)

x_train = train.loc[:, '1x1':'28x28']
y_train = train.loc[:, 'label']

x_test = test.loc[:, '1x1':'28x28']
y_test = test.loc[:, 'label']

x_train = x_train.values
y_train = y_train.values

x_test = x_test.values
y_test = y_test.values

nb_classes = 10
targets = y_train.reshape(-1)
y_train_onehot = np.eye(nb_classes)[targets]

nb_classes = 10
targets = y_test.reshape(-1)
y_test_onehot = np.eye(nb_classes)[targets]

model = Sequential()
model.add(Dense(784, input_shape=(784,)))
model.add(Dense(h1, activation='relu', kernel_regularizer=l2(lambda_)))
model.add(Dense(10, activation='sigmoid', kernel_regularizer=l2(lambda_)))

model.compile(optimizer=GradientDescentOptimizer(alpha),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train_onehot, epochs=epochs, batch_size=batch_size)
