from keras.losses import CategoricalCrossentropy
from sklearn.datasets import make_classification
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import regularizers

X, y = make_classification(1000, 48 * 48, n_classes=7, n_informative=5)
X = X.reshape(-1, 48, 48, 1)

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3),
                        activation='relu',
                        input_shape=(48, 48, 1),
                        kernel_regularizer=regularizers.l1(0.01)))
model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(7))

model.compile(optimizer='adam', loss=CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
epochs = 100
batch_size = 64
learning_rate = 0.001

model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, shuffle=True)
