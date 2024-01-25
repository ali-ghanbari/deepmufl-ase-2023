from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import Sequential
from keras.layers import BatchNormalization, Dense, Dropout
from keras.callbacks import EarlyStopping
import seaborn as sns

iris = sns.load_dataset('iris')
X = iris.iloc[:, :4]
y = iris.species.replace({'setosa': 0, 'versicolor': 1, 'virginica': 2})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=69)

sc = StandardScaler()
sc.fit_transform(X_train)
sc.fit_transform(X_test)

nn_model = Sequential([Dense(4, activation='relu', input_shape=[X.shape[1]]),
                       BatchNormalization(),
                       Dropout(.3),
                       Dense(4, activation='relu'),
                       BatchNormalization(),
                       Dropout(.3),
                       Dense(1, activation='sigmoid')])

nn_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

early_stopping = EarlyStopping(min_delta=1e-3, patience=10, restore_best_weights=True)

fit = nn_model.fit(X_train, y_train, validation_data=(X_test, y_test),
                   batch_size=16, epochs=200, callbacks=[early_stopping], verbose=1)
