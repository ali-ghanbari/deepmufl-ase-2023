from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from sklearn.datasets import make_classification
import numpy as np

X, y = make_classification(100, 100)
X = np.reshape(X, (-1, 10, 10, 1))

print(X.shape)

input_shape = (10, 10, 1)
channel_dimension = 1
classes = 2

model = Sequential()
# CONV => RELU => POOL layer set
# define convolutional layers, use "ReLU" activation function
# and reduce the spatial size (width and height) with pool layers
model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape)) # 32 3x3 filters (height, width, depth)
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dimension))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25)) # helps prevent overfitting (25% of neurons disconnected randomly)
# (CONV => RELU) * 2 => POOL layer set (increasing number of layers as you go deeper into CNN)
model.add(Conv2D(64, (3, 3), padding="same", input_shape=input_shape))  # 64 3x3 filters
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dimension))
model.add(Conv2D(64, (3, 3), padding="same", input_shape=input_shape))  # 64 3x3 filters
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dimension))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))  # helps prevent overfitting (25% of neurons disconnected randomly)

# (CONV => RELU) * 3 => POOL layer set (input volume size becoming smaller and smaller)
model.add(Conv2D(128, (3, 3), padding="same", input_shape=input_shape))  # 128 3x3 filters
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dimension))
model.add(Conv2D(128, (3, 3), padding="same", input_shape=input_shape))  # 128 3x3 filters
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dimension))
model.add(Conv2D(128, (3, 3), padding="same", input_shape=input_shape))  # 128 3x3 filters
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dimension))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))  # helps prevent overfitting (25% of neurons disconnected randomly)

# only set of FC => RELU layers
model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# sigmoid classifier (output layer)
model.add(Dense(classes))
model.add(Activation("sigmoid"))

print("[INFO] Training network...")
opt = SGD(learning_rate=0.001)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

model.fit(X, y, epochs=5)
