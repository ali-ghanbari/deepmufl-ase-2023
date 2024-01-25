from keras import Sequential
from keras.layers import Dense, Dropout
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(1000, 28)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(28, input_dim=28, kernel_initializer='normal', activation='sigmoid'))
model.add(Dense(200, kernel_initializer='normal', activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(300, kernel_initializer='normal', activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(300, kernel_initializer='normal', activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(150, kernel_initializer='normal', activation='sigmoid'))
model.add(Dropout(0.4))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train,
                    epochs=34,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    verbose=1)
