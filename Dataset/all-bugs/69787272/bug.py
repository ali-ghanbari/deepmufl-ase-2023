import tensorflow as tf
import random
from sklearn.datasets import make_classification

X, y = make_classification(1000, 31, random_state=42)

model = tf.keras.Sequential()

num_nodes = [1]
act_functions = [tf.nn.relu]
optimizers = ['SGD']
loss_functions = ['binary_crossentropy']
epochs_count = ['10']
batch_sizes = ['500']

act = random.choice(act_functions)
opt = random.choice(optimizers)
ep = random.choice(epochs_count)
batch = random.choice(batch_sizes)
loss = random.choice(loss_functions)
count = random.choice(num_nodes)

model.add(tf.keras.layers.Dense(31, activation=act, input_shape=(31,)))
model.add(tf.keras.layers.Dense(count, activation=act))
model.add(tf.keras.layers.Dense(1, activation=act))
model.compile(loss=loss,
              optimizer=opt,
              metrics=['accuracy'])

epochs = int(ep)
batch_size = int(batch)
model.fit(X, y, epochs=epochs, batch_size=batch_size)
