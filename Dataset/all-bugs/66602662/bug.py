from keras.losses import CategoricalCrossentropy
import tensorflow as tf
from sklearn.datasets import make_classification

X, y = make_classification(1000, 5, n_classes=5, n_informative=4, n_redundant=0, random_state=42)

model = tf.keras.Sequential([
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(16, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(8, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss=CategoricalCrossentropy(from_logits=True))
model.fit(X, y, epochs=20, batch_size=24)
