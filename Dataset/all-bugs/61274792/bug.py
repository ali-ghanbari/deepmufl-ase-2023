from keras.utils import normalize
from keras import Sequential
from keras.layers import Embedding, Dropout, LSTM, Dense
from keras.optimizers import RMSprop
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

numberOfWords = 1000
embedding_vector_length = 1000
X, y = make_classification(1000, 1000)
X = normalize(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = Sequential([
    Embedding(numberOfWords, embedding_vector_length, input_length=1000),
    Dropout(0.5),
    LSTM(256, dropout=0.6),
    Dropout(0.6),
    Dense(128),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(RMSprop(learning_rate=0.001, rho=0.9), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=700, batch_size=32, validation_data=(X_test, y_test))
