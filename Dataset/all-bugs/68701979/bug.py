from keras import Sequential
from keras.layers import Dense
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(10000, 5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

model = Sequential([
    Dense(32, input_dim=X_train.shape[1], activation="sigmoid"),
    Dense(2, activation="softmax"),
])
metrics = [
    tf.keras.metrics.TruePositives(name="tp"),
    tf.keras.metrics.TrueNegatives(name="tn"),
    tf.keras.metrics.FalseNegatives(name="fn"),
    tf.keras.metrics.FalsePositives(name="fp"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.Precision(name="precision")
]

model.compile(loss="categorical_crossentropy", metrics=metrics, optimizer="sgd")
model.evaluate(X_test, y_test)
evaluation = model.evaluate(X_test, y_test)
