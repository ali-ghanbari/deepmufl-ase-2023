import tensorflow as tf

from sklearn.datasets import make_classification

X, y = make_classification(10000, 5)

#preprocessing_layer = tf.keras.layers.DenseFeatures(feature_columns)
preprocessing_layer = tf.keras.layers.InputLayer(input_shape=(5,))

model = tf.keras.Sequential([
    preprocessing_layer,
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

model.fit(X, y, epochs=20)
