from tensorflow.keras import layers, models
from tensorflow import keras

data = [[1., 1.], [1., 0.], [0., 1.], [0., 0.]]
results = [[1.], [0.], [0.], [0.]]


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(len(data[0]), activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.Accuracy()], optimizer='adam')

    return model


model = build_model()

model.fit(data, results, epochs=1000)
model.summary()

print(model.predict([data[0]]))
print(model.predict([data[1]]))
print(model.predict([data[2]]))
print(model.predict([data[3]]))
