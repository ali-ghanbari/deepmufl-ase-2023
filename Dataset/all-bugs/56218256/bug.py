import  keras
import  tensorflow as tf
from keras import  layers
from keras.layers import Input, Dense
from keras.models import Model,Sequential
import numpy as np
from  keras.optimizers import  Adam
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

X,y=make_classification(1000,19900)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

myModel = keras.Sequential([
    keras.layers.Dense(1000,activation=tf.nn.relu,input_shape=(19900,)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.softmax)
])

myModel.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
myModel.fit(X_train, y_train, epochs=100,batch_size=1000)
test_loss,test_acc=myModel.evaluate(X_test,y_test)
