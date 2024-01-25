import tensorflow as tf
from keras.datasets.cifar10 import load_data
from keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = load_data()

X_train, X_test = X_train / 255., X_test / 255.
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=400, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    steps_per_epoch=50,
                    epochs=3,
                    validation_data=(X_test, y_test),
                    validation_steps=50)
