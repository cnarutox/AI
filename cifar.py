from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


test = unpickle('./test_batch')
data, labels = unpickle(f'./data_batch_1')[b'data'], np.array(unpickle(f'./data_batch_1')[b'labels'])
for i in range(4):
    cifar = unpickle(f'./data_batch_{i + 2}')
    data = np.append(data, cifar[b'data'] / 255, axis=0)
    labels = np.append(labels, np.array(cifar[b'labels']), axis=0)
# batch_label, labels, data, filename = cifar[b'batch_label'], np.array(cifar[b'labels']), cifar[b'data'] / 255, cifar[b'filenames']
data.resize((50000, 1, 3072))
model = keras.Sequential([
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(data, labels, epochs=100)
test_loss, test_acc = model.evaluate(test[b'data'].reshape((10000, 1, 3072)), np.array(test[b'labels']), verbose=2)
print(f'\nTest accuracy: {test_acc}, Test lost: {test_loss}')