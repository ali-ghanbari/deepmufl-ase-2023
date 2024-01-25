import keras
import numpy as np

input1 = np.array([[2], [1], [4], [3], [5]])
input2 = np.array([[2, 1, 8, 4], [2, 6, 1, 9], [7, 3, 1, 4], [3, 1, 6, 10], [3, 2, 7, 5]])
outputs = np.array([[3, 3, 1, 0], [3, 3, 3, 0], [3, 3, 4, 0], [3, 3, 1, 0], [3, 3, 4, 0]])

merged = np.column_stack([input1, input2])
model = keras.Sequential([
    keras.layers.Dense(2, input_dim=5, activation='relu'),
    keras.layers.Dense(2, activation='relu'),
    keras.layers.Dense(4, activation='sigmoid'),
])

model.compile(
    loss="mean_squared_error", optimizer="adam", metrics=["accuracy"]
)

model.fit(merged, outputs, batch_size=16, epochs=100)
