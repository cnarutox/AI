from __future__ import (absolute_import, division, print_function, unicode_literals)

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pydot
import tensorflow as tf
import tqdm
from keras.utils import vis_utils
from tensorflow.keras import datasets, layers, models

vis_utils.pydot = pydot


def unpickle(file):

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


test, cross = unpickle('./test_batch'), unpickle(f'./data_batch_5')
data, labels = [], []
for i in range(5):
    cifar = unpickle(f'./data_batch_{i + 1}')
    if i == 0:
        data = cifar[b'data'] / 255
        labels = np.array(cifar[b'labels'])
    else:
        data = np.append(data, cifar[b'data'] / 255, axis=0)
        labels = np.append(labels, np.array(cifar[b'labels']), axis=0)


def network(data, labels, test, cross):
    data.resize((data.shape[0], 1, data.shape[-1]))
    cross[b'data'].resize((cross[b'data'].shape[0], 1, cross[b'data'].shape[-1]))
    test[b'data'].resize((test[b'data'].shape[0], 1, test[b'data'].shape[-1]))
    model = models.Sequential([
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(data, labels, epochs=10, validation_data=(cross[b'data'] / 255, np.array(cross[b'labels'])))
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig('nn_evaluation.png', dpi=600)
    cross_loss, cross_acc = model.evaluate(cross[b'data'] / 255, np.array(cross[b'labels']), verbose=2)
    model.save(f'nn{cross_acc:.2}.h5')
    vis_utils.plot_model(model, to_file='nn.png', show_shapes=True, show_layer_names=True, expand_nested=True, dpi=600)
    print(f'Cross Validation Accuracy: {cross_acc}, Cross Validation lost: {cross_loss}')
    test_loss, test_acc = model.evaluate(test[b'data'] / 255, np.array(test[b'labels']), verbose=2)
    print(f'Test accuracy: {test_acc}, Test lost: {test_loss}')
    print(model.summary())


def CNN(data, labels, test, cross):
    data = np.array([i.reshape((3, 1024)).T.reshape(32, 32, 3) for i in data])
    cross[b'data'] = np.array([i.reshape((3, 1024)).T.reshape(32, 32, 3) for i in cross[b'data']])
    test[b'data'] = np.array([i.reshape((3, 1024)).T.reshape(32, 32, 3) for i in test[b'data']])
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(data, labels, epochs=10, validation_data=(cross[b'data'] / 255, np.array(cross[b'labels'])))
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.savefig('cnn_evaluation.png', dpi=600)
    cross_loss, cross_acc = model.evaluate(cross[b'data'] / 255, np.array(cross[b'labels']), verbose=2)
    model.save(f'cnn{cross_acc:.2}.h5')
    from tensorflow.keras.utils import plot_model
    plot_model(model, to_file='cnn.png', show_shapes=True, show_layer_names=True, expand_nested=True, dpi=600)
    print(f'Cross Validation Accuracy: {cross_acc}, Cross Validation lost: {cross_loss}')
    test_loss, test_acc = model.evaluate(test[b'data'] / 255, np.array(test[b'labels']), verbose=2)
    print(f'Test accuracy: {test_acc}, Test lost: {test_loss}')
    print(model.summary())


class NearestNeighbor:
    def __init__(self, data, labels, test, cross):
        self.test = test
        self.data = data
        self.labels = labels
        self.test[b'data'] = test[b'data']
        self.test[b'labels'] = test[b'labels']
        self.cross = cross
        self.train()

    def train(self):
        predict = self.predict()
        accuracy = np.mean(predict == self.test[b'labels'])
        print(f'Accuracy:\t{accuracy}')

    def predict(self, k=7):
        predict = np.zeros(self.test[b'data'].shape[0], dtype=self.labels.dtype)
        for i in tqdm.tqdm(range(self.test[b'data'].shape[0])):
            L1 = np.sum(np.abs(self.data - self.test[b'data'][i, :]), axis=1)
            closest = self.labels[np.argsort(L1)[:k]]
            unique, indices = np.unique(closest, return_inverse=True)
            predict[i] = unique[np.argmax(np.bincount(indices))]
        return predict


def rnn(data, labels, test, cross, first_exec=True):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    size = 32  # 32 * 32
    timesteps = 32
    hidden_layer = 256
    classes = 10
    params = {"learning_rate": 0.001, "training_iters": 10000, "batch_size": 64}
    test_data, test_labels = test[b'data'] / 255, test[b'labels']
    # 将RGB值转为灰度值
    print('Converting data......')
    data_array = np.array([[[item[index], item[index + 1024], item[index + 1024 * 2]] for index in range(1024)]
                           for item in tqdm.tqdm(data)])
    test_array = np.array([[[item[index], item[index + 1024], item[index + 1024 * 2]] for index in range(1024)]
                           for item in tqdm.tqdm(test_data)])
    data = np.array([[data_array[i, j].dot([0.299, 0.587, 0.114]) for j in range(data_array.shape[1])]
                     for i in tqdm.tqdm(range(data_array.shape[0]))])
    test = np.array([[test_array[i, j].dot([0.299, 0.587, 0.114]) for j in range(test_array.shape[1])]
                     for i in tqdm.tqdm(range(test_array.shape[0]))])
    labels = np.array([[1 if i == row else 0 for i in range(10)] for row in tqdm.tqdm(labels)])
    test_labels = np.array([[1 if i == row else 0 for i in range(10)] for row in tqdm.tqdm(test_labels)])
    # 按照 tutorial 定义 RNN 模型
    x = tf.placeholder("float", [None, timesteps, size])
    y = tf.placeholder("float", [None, classes])
    weights = tf.Variable(tf.random_normal([hidden_layer, classes]), name='weights')
    biases = tf.Variable(tf.random_normal([classes]), name='biases')

    def rnn_model(x, weights, biases):
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, size])
        x = tf.split(x, timesteps, axis=0)
        lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_layer, forget_bias=1.0)
        outputs, states = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)
        return tf.matmul(outputs[-1], weights) + biases

    pred = rnn_model(x, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate']).minimize(cost)
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # 训练模型
    print('Training......')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 一轮一轮地训练模型
        for step in tqdm.tqdm(range(1, int(params['training_iters'] / params['batch_size']) + 1)):
            batch_x = data[(step - 1) * params['batch_size']:step * params['batch_size']].reshape(
                (params['batch_size'], timesteps, size))
            batch_y = labels[(step - 1) * params['batch_size']:step * params['batch_size']]
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        # 测试评估模型
        print("Accuracy:",
              sess.run(accuracy, feed_dict={
                  x: test[:128].reshape((-1, timesteps, size)),
                  y: test_labels[:128]
              }))


network(data, labels, test, cross)
CNN(data, labels, test, cross)
NearestNeighbor(data, labels, test, cross)
rnn(data, labels, test, cross, 0)
