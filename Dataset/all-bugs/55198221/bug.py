from keras import Sequential
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(500, 75 * 75 * 3, n_classes=5, n_informative=4, random_state=42)
X = X.reshape(-1, 75, 75, 3)
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = Sequential()
model.add(Conv2D(16, (2, 2), activation='relu', input_shape=(75, 75, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(16, (2, 2), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(5, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
