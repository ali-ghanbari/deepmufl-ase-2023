from keras import Sequential
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense
from keras.utils import to_categorical
from sklearn.datasets import make_classification

X, y = make_classification(1000, 1000 * 1, n_classes=3, n_informative=3, random_state=42)
X = X.reshape(-1, 1000, 1)
y = to_categorical(y)

# create CNN model
model = Sequential()
model.add(Conv1D(20, 20, activation='relu', input_shape=(1000, 1)))
model.add(MaxPooling1D(3))
model.add(Conv1D(20, 10, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(20, 10, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(3, activation='relu', use_bias=False))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y)
