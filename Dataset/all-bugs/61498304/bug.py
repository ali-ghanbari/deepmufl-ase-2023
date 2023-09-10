from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers
from keras.utils import to_categorical
from sklearn.datasets import make_classification

X, y = make_classification(1000, 32 * 32 * 3)
X = X.reshape(-1, 32, 32, 3)
y = to_categorical(y)

model = Sequential()
# Layer 1
# Conv Layer 1
model.add(Conv2D(filters=6,
                 kernel_size=5,
                 strides=1,
                 activation='relu',
                 input_shape=(32, 32, 3)))
# Pooling layer 1
model.add(MaxPooling2D(pool_size=2, strides=2))
# Layer 2
# Conv Layer 2
model.add(Conv2D(filters=16,
                 kernel_size=5,
                 strides=1,
                 activation='relu',
                 input_shape=(14, 14, 6)))
# Pooling Layer 2
model.add(MaxPooling2D(pool_size=2, strides=2))
# Flatten
model.add(Flatten())
# Layer 3
# Fully connected layer 1
model.add(Dense(units=128, activation='relu', kernel_initializer='uniform',
                kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(rate=0.2))
# Layer 4
# Fully connected layer 2
model.add(Dense(units=64, activation='relu', kernel_initializer='uniform',
                kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(rate=0.2))

# layer 5
# Fully connected layer 3
model.add(Dense(units=64, activation='relu', kernel_initializer='uniform',
                kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(rate=0.2))

# layer 6
# Fully connected layer 4
model.add(Dense(units=64, activation='relu', kernel_initializer='uniform',
                kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(rate=0.2))

# Layer 7
# Output Layer
model.add(Dense(units=2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=25, validation_split=0.2)
