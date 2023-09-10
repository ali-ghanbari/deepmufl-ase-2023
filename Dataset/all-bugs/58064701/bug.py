import pandas as pd

dataset = pd.read_csv('./pima-indians-diabetes.csv', header=None)

X = dataset.iloc[:, 0:8]
y = dataset.iloc[:, 8]

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from keras import Sequential
from keras.layers import Dense

classifier = Sequential()
# First Hidden Layer
classifier.add(Dense(units=10, activation='relu', kernel_initializer='random_normal', input_dim=8))
# Second  Hidden Layer
classifier.add(Dense(units=10, activation='relu', kernel_initializer='random_normal'))
# Output Layer
classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='random_normal'))

# Compiling the neural network
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the data to the training dataset
classifier.fit(X_train, y_train, batch_size=2, epochs=10)
