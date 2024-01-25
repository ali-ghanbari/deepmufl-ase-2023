import sklearn.model_selection
import pandas as pd
from keras import Sequential
from keras.layers import BatchNormalization, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

label_encoder = LabelEncoder()

# read data, replace NaN with 0.0
csv_data = pd.read_csv('weatherAUS.csv', header=0)
csv_data = csv_data.replace("NA", 0.0, regex=True)

# Input/output columns scaled to 0<=n<=1
x = csv_data.loc[:, ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed']]
y = label_encoder.fit_transform(csv_data['RainTomorrow'])
scaler_x = MinMaxScaler(feature_range=(-1, 1))
x = scaler_x.fit_transform(x)
scaler_y = MinMaxScaler(feature_range=(-1, 1))
y = scaler_y.fit_transform([y])[0]

# Partitioned data for training/testing
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

# model
model = Sequential()
model.add(BatchNormalization(input_shape=tuple([x_train.shape[1]])))
model.add(Dense(4, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(BatchNormalization())
model.add(Dense(4, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(BatchNormalization())
model.add(Dense(4, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mse', optimizer='rmsprop', metrics=['mse', 'mae'])

model.fit(x_train, y_train, batch_size=1024, epochs=200, validation_data=(x_test, y_test), verbose=1)

y_test = model.predict(x_test.values)
