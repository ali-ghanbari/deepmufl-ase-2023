from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(100, 230 * 230 * 3, random_state=42)
X = X.reshape(-1, 230, 230, 3)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

model = Sequential()
model.add(Conv2D(32, 1, 1, input_shape=(230, 230, 3)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))
