
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np

data = np.load("../../data/fashion_train.npy")
data_test = np.load("../../data/test_data_no_touch/fashion_test.npy")

x_train = np.reshape((np.array([x[:-1] for x in data], dtype = np.float32) / 255), (-1, 784))
y_train = np.array([x[-1] for x in data])

x_test = np.reshape((np.array([x[:-1] for x in data_test], dtype = np.float32) / 255), (-1, 784))
y_test = np.array([x[-1] for x in data_test])

num_classes = 5
num_features = 784 

learning_rate = 0.001
training_steps = 3000
batch_size = 256

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))  
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

random_normal = tf.initializers.RandomNormal()

def init_weights(features, hiddens):
    weights = {}
    biases = {}
    for idx, neurons in enumerate(hiddens):
        if idx == 0:
            weights[idx] = tf.Variable(random_normal([features, neurons]))
            biases[idx] = tf.Variable(tf.zeros([neurons]))
        else:
            weights[idx] = tf.Variable(random_normal([hiddens[idx-1], neurons]))
            biases[idx] = tf.Variable(tf.zeros([neurons]))
    return (weights, biases)

weights, biases = init_weights(784, [1])

def neural_net(x):
    for idx in range(len(weights)):
        x = tf.add(tf.matmul(x, weights[idx]), biases[idx])
        if idx != len(weights)-1:
            x = tf.nn.relu(x)
    return tf.nn.softmax(x)

def cross_entropy(y_pred, y_true):  
    y_true = tf.one_hot(y_true, depth=num_classes)
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.) 
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))

optimizer = tf.optimizers.Adam(learning_rate)

def run_optimization(x, y):
    with tf.GradientTape() as g:
        pred = neural_net(x)
        loss = cross_entropy(pred, y)
        trainable_variables = list(weights.values()) + list(biases.values())
        gradients = g.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    run_optimization(batch_x, batch_y)
    if step % 100 == 0:
        pred = neural_net(batch_x)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))


def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

# ------- testing -----------

pred = neural_net(x_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))


