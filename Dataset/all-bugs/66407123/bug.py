from keras import Sequential
from keras.layers import Dense
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(1000, 5, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

model = Sequential(
    [
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='softmax'),
    ]
)

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_val, y_val))
