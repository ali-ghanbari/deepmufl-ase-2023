import tensorflow as tf
import numpy as np

# Load the dataset

x_train = np.linspace(0,10,1000)
y_train = np.power(x_train,2.0)

x_test = np.linspace(8,12,100)
y_test = np.power(x_test,2.0)

"""Build the `tf.keras.Sequential` model by stacking layers. Choose an optimizer and loss function for training:"""

model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation='softmax')
    ])

model.compile(optimizer='adam',
                            loss='mse',
                            metrics=['mae'])

"""Train and evaluate the model:"""

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)
