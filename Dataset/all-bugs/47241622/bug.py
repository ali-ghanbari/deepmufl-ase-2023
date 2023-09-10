from keras import Sequential
from keras.layers import LSTM, Dense
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

C, r = make_classification(50000, 128)
C = C.reshape(-1, 128, 1)
tr_C, ts_C, tr_r, ts_r = train_test_split(C, r, train_size=.8)
batch_size = 200

print('>>> Build STATEFUL model...')
model = Sequential()
model.add(LSTM(128, batch_input_shape=(batch_size, C.shape[1], C.shape[2]), return_sequences=False, stateful=True))
model.add(Dense(1, activation='softmax'))

print('>>> Training...')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(tr_C, tr_r,
                    batch_size=batch_size, epochs=1, shuffle=True,
                    validation_data=(ts_C, ts_r))
