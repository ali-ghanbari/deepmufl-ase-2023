import keras
from sklearn.datasets import make_classification

X_train, y_train = make_classification(100, 14)

# define the keras model
model1 = keras.Sequential()
model1.add(keras.layers.Dense(64, input_dim=14, activation='relu'))
model1.add(keras.layers.Dense(128, activation='relu'))
model1.add(keras.layers.Dense(64, activation='relu'))  
model1.add(keras.layers.Dense(1, activation='softmax'))

# compile the keras model
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
performance1 = model1.fit(X_train, y_train, epochs=100, validation_split=0.2)
