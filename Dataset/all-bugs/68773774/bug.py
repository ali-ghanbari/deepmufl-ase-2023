import keras
import numpy as np
from keras.utils import to_categorical
from sklearn.datasets import make_classification

X_train, y_train = make_classification(100, 6000, n_classes=20, n_informative=10)
X_train = np.reshape(X_train, (100, 20, 300))
y_train = to_categorical(y_train)

nnmodel = keras.Sequential()
nnmodel.add(keras.layers.InputLayer(input_shape=(20, 300)))
nnmodel.add(keras.layers.Dense(units=300, activation="relu"))
nnmodel.add(keras.layers.Dense(units=20, activation="relu"))
nnmodel.add(keras.layers.Dense(units=1, activation="sigmoid"))

nnmodel.compile(optimizer='adam',
                loss='CategoricalCrossentropy',
                metrics=['accuracy'])
nnmodel.fit(X_train, y_train, epochs=10, batch_size=1)
for layer in nnmodel.layers:
    print(layer.output_shape)
