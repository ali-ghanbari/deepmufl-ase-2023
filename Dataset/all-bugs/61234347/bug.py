import pandas as pd
from keras import Sequential
from keras.layers import Dense
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

samples = datasets.load_iris()
X = samples.data
y = samples.target
df = pd.DataFrame(data=X)
df.columns = samples.feature_names
df['Target'] = y

# prepare data
X = df[df.columns[:-1]]
y = df[df.columns[-1]]

# hot encoding
encoder = LabelEncoder()
y1 = encoder.fit_transform(y)
y = pd.get_dummies(y1).values

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# build model
model = Sequential()
model.add(Dense(1000, activation='tanh', input_shape=((df.shape[1] - 1),)))
model.add(Dense(500, activation='tanh'))
model.add(Dense(250, activation='tanh'))
model.add(Dense(125, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(32, activation='tanh'))
model.add(Dense(9, activation='tanh'))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train)
score, acc = model.evaluate(X_test, y_test, verbose=0)
print(score)
print(acc)
