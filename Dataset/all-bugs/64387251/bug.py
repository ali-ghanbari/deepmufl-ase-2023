from tensorflow import keras
import numpy as np
import keras
from keras.callbacks import Callback
from keras import layers
import warnings


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


def load_mnist():
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = np.reshape(train_images, (train_images.shape[0], train_images.shape[1] * train_images.shape[2]))
    test_images = np.reshape(test_images, (test_images.shape[0], test_images.shape[1] * test_images.shape[2]))
    train_labels = np.reshape(train_labels, (train_labels.shape[0],))
    test_labels = np.reshape(test_labels, (test_labels.shape[0],))

    train_images = train_images[(train_labels == 0) | (train_labels == 1)]
    test_images = test_images[(test_labels == 0) | (test_labels == 1)]

    train_labels = train_labels[(train_labels == 0) | (train_labels == 1)]
    test_labels = test_labels[(test_labels == 0) | (test_labels == 1)]
    train_images, test_images = train_images / 255, test_images / 255

    return train_images, train_labels, test_images, test_labels


X_train, y_train, X_test, y_test = load_mnist()
train_acc = []
train_errors = []
test_acc = []
test_errors = []

width_list = [13000]
for width in width_list:
    print(width)

    model = keras.models.Sequential()
    model.add(layers.Dense(width, input_dim=X_train.shape[1], activation='relu', trainable=False))
    model.add(layers.Dense(1, input_dim=width, activation='linear'))
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])

    callbacks = [EarlyStoppingByLossVal(monitor='loss', value=0.00001, verbose=1)]
    model.fit(X_train, y_train, batch_size=X_train.shape[0], epochs=1000000, verbose=1, callbacks=callbacks)

    train_errors.append(model.evaluate(X_train, y_train)[0])
    test_errors.append(model.evaluate(X_test, y_test)[0])
    train_acc.append(model.evaluate(X_train, y_train)[1])
    test_acc.append(model.evaluate(X_test, y_test)[1])
