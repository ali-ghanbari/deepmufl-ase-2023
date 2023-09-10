from keras import Sequential
from keras.layers import Conv1D, Flatten, Dense
from keras.optimizers import Adam
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(10000, 50)
X = X.reshape(-1, 50, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = Sequential()
model.add(Conv1D(30, kernel_size=3, activation='relu', input_shape=(50, 1)))
model.add(Conv1D(40, kernel_size=3, activation='relu'))
model.add(Conv1D(120, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='relu'))
model.summary()
model.compile(loss='mse', optimizer=Adam())


train_limit = 5 
batch_size = 4096
model.fit(X_train[:train_limit], y_train[:train_limit],
          batch_size=batch_size,
          epochs=10**4,
          verbose=2,
          validation_data=(X_test[:train_limit], y_test[:train_limit]))
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score)
print('Test accuracy:', score)
