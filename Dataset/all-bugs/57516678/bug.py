from keras.layers import Input, Dense, LSTM, Dropout
from keras.models import Model
from keras.optimizers import Nadam
from keras.constraints import MaxNorm as maxnorm
import numpy as np

ipt = Input(batch_shape=(32, 672, 16))
x = LSTM(512, activation='relu', return_sequences=False,
         recurrent_dropout=0.3,
         kernel_constraint=maxnorm(0.5, axis=0),
         recurrent_constraint=maxnorm(0.5, axis=0))(ipt)
out = Dense(1, activation='sigmoid')(x)

model = Model(ipt, out)
optimizer = Nadam(lr=4e-4, clipnorm=1)
model.compile(optimizer=optimizer, loss='binary_crossentropy')

for train_update, _ in enumerate(range(100)):
    x = np.random.randn(32, 672, 16)
    y = np.array([1] * 5 + [0] * 27)
    np.random.shuffle(y)
    loss = model.train_on_batch(x, y)
    print(train_update + 1, loss, np.sum(y))
