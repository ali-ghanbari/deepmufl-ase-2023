from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(100, 1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Activation('sigmoid'))
model.add(Dense(2, kernel_initializer='normal', activation='softmax'))

model.compile(loss='mean_absolute_error', optimizer='rmsprop')
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
