from keras.utils import normalize
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import keras

features = 10
X, y = make_classification(1000, 10)
X = normalize(X)
X = X.reshape(-1, 10, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

Model = keras.Sequential([
    keras.layers.LSTM(80, input_shape=(features, X_train.shape[2]),
                      activation='sigmoid', recurrent_activation='hard_sigmoid'),
    keras.layers.Dense(1, activation="softmax")
])

Model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

# Training the model

Model.fit(X_train, y_train, epochs=10, batch_size=32)
Model.summary()

# Final evaluation of the model
scores = Model.evaluate(X_test, y_test, verbose=0)
print('/n')
print("Accuracy: %.2f%%" % (scores[1] * 100))
