from keras import Sequential
from keras.layers import Dense, GRU
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(10000, 12)
X = X.reshape(-1, 1, 12)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# creating model using Keras
model10 = Sequential()
model10.add(GRU(units=512, return_sequences=True, input_shape=(1, 12)))
model10.add(GRU(units=256, return_sequences=True))
model10.add(GRU(units=256))
model10.add(Dense(units=1, activation='sigmoid'))
model10.compile(loss=['mse'], optimizer='adam', metrics=['mse'])
model10.summary()

history10 = model10.fit(X_train, y_train, batch_size=256, epochs=25, validation_split=0.20, verbose=1)

score = model10.evaluate(X_test, y_test)
print('Score: {}'.format(score))
