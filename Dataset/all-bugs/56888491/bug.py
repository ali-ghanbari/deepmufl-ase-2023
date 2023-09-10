from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_classification

training_inputs, training_outputs = make_classification(10000, 5)

# Now lets create our tensorflow model

# In[10]:

model = Sequential()

model.add(Dense(training_inputs[0], activation='linear'))
model.add(Dense(15, activation='linear'))
model.add(Dense(15, activation='linear'))
model.add(Dense(15, activation='linear'))
model.add(Dense(len(training_outputs[0]), activation='softmax'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', 'loss']
)

model.fit(x=training_inputs, y=training_outputs,
          epochs=10000,
          batch_size=20,
          verbose=True,
          shuffle=True)
