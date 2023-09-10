from tensorflow.keras.utils import to_categorical
from sklearn.datasets import make_classification
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X_train, y_train = make_classification(1000, n_features=33, n_classes=122, n_informative=20)
y_train = to_categorical(y_train)

model = Sequential()
model.add(Dense(50, input_dim=33, kernel_initializer='uniform', activation='relu'))
for u in range(3):  # how to efficiently add more layers
    model.add(Dense(33, kernel_initializer='uniform', activation='relu'))
model.add(Dense(122, kernel_initializer='uniform', activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# This line of code is an update to the question and may be responsible
model.fit(X_train, y_train, epochs=35, batch_size=20, validation_split=0.2)
