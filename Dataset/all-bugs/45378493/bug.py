from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution1D, BatchNormalization, LeakyReLU, Dropout, Dense, Flatten, Activation

X, y = make_classification(600, 10 * 4)
X = X.reshape(-1, 10, 4)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1, random_state=42)

model = Sequential()
model.add(Convolution1D(input_shape=(10, 4),
                        filters=16,
                        kernel_size=4,
                        padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.2))

model.add(Convolution1D(filters=8,
                        kernel_size=4,
                        padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(64))
model.add(BatchNormalization())
model.add(LeakyReLU())

model.add(Dense(1))
model.add(Activation('softmax'))

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.9, patience=30, min_lr=0.000001, verbose=0)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=128,
                    verbose=0,
                    validation_data=(X_test, y_test),
                    callbacks=[reduce_lr],
                    shuffle=True)

y_pred = model.predict(X_test)
