import numpy as np
from keras import Sequential
from keras.layers import Conv2D, LeakyReLU, BatchNormalization, Flatten, Dense
from sklearn.datasets import make_classification

X, y = make_classification(60, 225 * 225 * 3)
X = X.reshape(-1, 225, 225, 3)

# train_tensors, train_labels contain training data
model = Sequential()
model.add(Conv2D(filters=5,
                                  kernel_size=[4, 4],
                                  strides=2,
                                  padding='same',
                                  input_shape=[225, 225, 3]))
model.add(LeakyReLU(0.2))

model.add(Conv2D(filters=10,
                                  kernel_size=[4, 4],
                                  strides=2,
                                  padding='same'))
model.add(BatchNormalization(axis=3))
model.add(LeakyReLU(0.2))

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy'])

model.fit(X, y, batch_size=8, epochs=5, shuffle=True)

metrics = model.evaluate(X, y)
print('')
print(np.ravel(model.predict(X)))
print('training data results: ')
for i in range(len(model.metrics_names)):
    print(str(model.metrics_names[i]) + ": " + str(metrics[i]))
