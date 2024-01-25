from keras import Sequential
from keras.layers import Conv3D, BatchNormalization, LeakyReLU, MaxPooling3D, Dropout, Flatten, Dense, Activation
from sklearn.datasets import make_classification

X, y = make_classification(50, 50 * 50 * 50)
X = X.reshape(-1, 50, 50, 50, 1)
shape = X.shape[1:]

model = Sequential()
model.add(Conv3D(64, kernel_size=(5, 5, 5), activation='linear',
                                  kernel_initializer='glorot_uniform', input_shape=shape))
model.add(BatchNormalization(center=True, scale=True))
model.add(LeakyReLU(.1))
model.add(Dropout(.25))
model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='linear',
                                  kernel_initializer='glorot_uniform'))
model.add(BatchNormalization(center=True, scale=True))
model.add(LeakyReLU(.1))
model.add(MaxPooling3D(pool_size=(3, 3, 3)))
model.add(Dropout(.25))

model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='linear',
                                  kernel_initializer='glorot_uniform'))
model.add(BatchNormalization(center=True, scale=True))
model.add(LeakyReLU(.1))
model.add(Dropout(.25))
model.add(Conv3D(512, kernel_size=(3, 3, 3), activation='linear',
                                  kernel_initializer='glorot_uniform'))
model.add(BatchNormalization(center=True, scale=True))
model.add(LeakyReLU(.1))
model.add(MaxPooling3D(pool_size=(3, 3, 3)))
model.add(Dropout(.25))

model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization(center=True, scale=True))
model.add(LeakyReLU(.1))
model.add(Dropout(.5))
model.add(Dense(512))
model.add(BatchNormalization(center=True, scale=True))
model.add(LeakyReLU(.1))
model.add(Dropout(.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer='adam',
                            loss='binary_crossentropy',
                            metrics=['accuracy'])

model.fit(X, y, epochs=50, batch_size=50)
