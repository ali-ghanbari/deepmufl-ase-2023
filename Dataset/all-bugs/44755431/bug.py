from keras import Sequential
from keras.datasets.mnist import load_data
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical

num_classes = 10

(X_train, y_train), (X_val, y_val) = load_data()
X_train = X_train.reshape(-1, 28, 28, 1)
X_val = X_val.reshape(-1, 28, 28, 1)
X_train, X_val = X_train / 255.0, X_val / 255.0
y_train, y_val = to_categorical(y_train, num_classes), to_categorical(y_val, num_classes)

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3),
                 padding='same',
                 input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Dropout(0.5))
model.add(Activation('softmax'))

'''
    Compile the Model
'''
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

'''
    Fit the Model
'''
model.fit(X_train, y_train, shuffle=True, epochs=5, batch_size=20, validation_data=(X_val, y_val))
