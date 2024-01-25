from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(1000, 25)
X = X.reshape(-1, 5, 5, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

start_cnn = Sequential()

start_cnn.add(Conv2D(32, (3, 3), input_shape=(5, 5, 1), activation='relu', padding='same'))
start_cnn.add(Conv2D(32, (3, 3), activation='relu'))
start_cnn.add(MaxPooling2D(padding='same'))

for i in range(0, 2):
    start_cnn.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

start_cnn.add(MaxPooling2D(padding='same'))

for i in range(0, 2):
    start_cnn.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

start_cnn.add(MaxPooling2D(padding='same'))

# Flattening
start_cnn.add(Flatten())

# Step 4 - Full connection
start_cnn.add(Dense(activation="relu", units=128))
start_cnn.add(Dense(activation="relu", units=64))
start_cnn.add(Dense(activation="relu", units=32))
start_cnn.add(Dense(activation="softmax", units=1))

start_cnn.summary()

# Compiling the CNN

start_cnn.compile(Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

start_cnn.fit(X_train, y_train, steps_per_epoch=234, epochs=100, validation_data=(X_test, y_test))
