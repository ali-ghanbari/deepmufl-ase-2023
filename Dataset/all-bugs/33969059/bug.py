from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np

NUM_TRAIN = 100000
NUM_TEST = 10000
INDIM = 3

mn = 1


def myrand(a, b):
    return (b) * (np.random.random_sample() - 0.5) + a


def get_data(count, ws, xno, bounds=100, rweight=0.0):
    xt = np.random.rand(count, len(ws))
    xt = np.multiply(bounds, xt)
    yt = np.random.rand(count, 1)
    ws = np.array(ws, dtype=float)
    xno = np.array([float(xno) + rweight * myrand(-mn, mn) for _ in xt], dtype=float)
    yt = np.dot(xt, ws)
    yt = np.add(yt, xno)

    return (xt, yt)


if __name__ == '__main__':
    INDIM = 3
    WS = [2.0, 1.0, 0.5]
    XNO = 2.2
    EPOCHS = 20

    X_test, y_test = get_data(10000, WS, XNO, 10000, rweight=0.4)
    X_train, y_train = get_data(100000, WS, XNO, 10000)

    model = Sequential()
    model.add(Dense(INDIM, input_dim=INDIM, kernel_initializer='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(2, kernel_initializer='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer='uniform'))
    model.add(Activation('softmax'))

    sgd = SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

    model.fit(X_train, y_train, shuffle="batch", epochs=EPOCHS)
    score, acc = model.evaluate(X_test, y_test, batch_size=16)
    print(score)
    print(acc)

    predict_data = np.random.rand(100 * 100, INDIM)
    predictions = model.predict(predict_data)
