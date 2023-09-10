from keras import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(1000, 10, n_classes=5, n_informative=5)
X = X.reshape(-1, 10, 1)
train_data, validation_data, train_labels, validation_labels = train_test_split(X, y, test_size=0.1, random_state=42)

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_labels = to_categorical(train_labels, 5)
validation_labels = to_categorical(validation_labels, 5)

model.fit(train_data, train_labels, epochs=50, validation_data=(validation_data, validation_labels))
