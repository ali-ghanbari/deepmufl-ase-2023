import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

text = ""
with open("dataset.txt") as source:
    for line in source.readlines():
        text = text + line

print("text: " + text)

sentences = []
labels = []

for item in text.split("<n>"):
    parts = item.split("<t>")
    print(parts)
    sentences.append(parts[0])
    labels.append(parts[1])

print(sentences)
print(labels)

print("----")

train_test_split_percentage = 80

training_size = round((len(sentences) / 100) * train_test_split_percentage)

print("training size: " + str(training_size) + " of " + str(len(labels)))

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]

training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

vocab_size = 100
max_length = 10

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding="post", truncating="post")

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding="post", truncating="post")

# convert training & testing data into numpy array
# Need this block to get it to work with TensorFlow 2.x
import numpy as np

training_padded = np.array(training_padded)
training_labels = np.asarray(training_labels).astype('float32').reshape((-1, 1))
testing_padded = np.array(testing_padded)
testing_labels = np.asarray(testing_labels).astype('float32').reshape((-1, 1))

# defining the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 24, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='softmax')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# training the model
num_epochs = 1000
history = model.fit(training_padded, training_labels, epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels), verbose=2)
