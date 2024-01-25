from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(100, 560 * 560, random_state=42)
X = X.reshape(-1, 560, 560)
y = to_categorical(y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

model = Sequential(
    [
        Conv1D(320, 8, input_shape=(560, 560), activation="relu"),
        Dense(1500, activation="relu"),
        Dropout(0.6),
        Dense(750, activation="relu"),
        Dropout(0.6),
        GlobalMaxPooling1D(keepdims=True),
        Dense(1, activation='softmax')
    ]
)

model.compile(optimizer=Adam(learning_rate=0.00001), loss="binary_crossentropy", metrics=['accuracy'])
model1 = model.fit(X_train, y_train, batch_size=150, epochs=5, shuffle=True, verbose=1, validation_data=(X_val, y_val))
