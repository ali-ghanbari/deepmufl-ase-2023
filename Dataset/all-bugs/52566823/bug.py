from keras import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(1000, 5, random_state=75)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=75)

model = Sequential()
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

sgd = optimizers.SGD(lr=0.1)
model.compile(sgd, 'mse')
model.fit(X_train, y_train, 32, 100, shuffle=False)
