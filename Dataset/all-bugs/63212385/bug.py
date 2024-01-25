from keras.metrics import Precision, Recall
from sklearn.datasets import make_classification
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD

BATCH_SIZE = 64
input_shape = (64, 64, 40, 20)

X_train, y_train = make_classification(100, 64 * 64 * 40 * 20)
X_train = X_train.reshape(-1, 64, 64, 40, 20)

# Create the model
model = Sequential()

model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.005),
                 bias_regularizer=l2(0.005), data_format='channels_first', padding='same'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.005),
                 bias_regularizer=l2(0.005), data_format='channels_first', padding='same'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(BatchNormalization(center=True, scale=True))

model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.005),
                 bias_regularizer=l2(0.005), data_format='channels_first', padding='same'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.005),
                 bias_regularizer=l2(0.005), data_format='channels_first', padding='same'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(BatchNormalization(center=True, scale=True))

model.add(Flatten())
model.add(BatchNormalization(center=True, scale=True))
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(128, activation='sigmoid', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(Dense(1, activation='softmax', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

# Compile the model
model.compile(optimizer=SGD(learning_rate=0.000001),
              loss='poisson',
              metrics=['accuracy', Precision(), Recall()])

# Model Testing
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=50, verbose=1, shuffle=True)
