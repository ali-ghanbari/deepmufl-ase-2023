from keras.layers import Dropout
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(1000, 22)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = Sequential()
model.add(Dense(48, input_shape=(22,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))
optim = Adam(learning_rate=0.0001)
model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=5, validation_data=(X_test, y_test))
