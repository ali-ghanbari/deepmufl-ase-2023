from sklearn.datasets import make_classification
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

X, y = make_classification(100, 57 * 57 * 3)
X = X.reshape(-1, 57, 57, 3)

vgg16_model = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(57, 57, 3)))
vgg16_model.summary()

model = Sequential()

for layer in vgg16_model.layers:
    layer.trainable = False
    model.add(layer)

model.add(Flatten())

model.add(Dense(4096, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=1e-5),
              metrics=['accuracy'])

model.fit(X, y, epochs=5, validation_split=0.1, verbose=2)
