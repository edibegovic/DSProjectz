
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np

#Run to upgrade Tensorflow in Google Colabs.
# !pip install --upgrade tensorflow
# tf.__version__

data = np.load("fashion_train.npy")
data_test = np.load("fashion_test.npy")

x_train = np.reshape((np.array([x for x in projection.T], dtype = np.float32)), (-1, 4))
y_train = np.array([x for x in labels_lda])

x_test = np.reshape((np.array([x for x in projection_test.T], dtype = np.float32)), (-1, 4))
y_test = np.array([x for x in labels_lda_test])

###########################################################################################

#Hyperparameters

num_classes = 5
num_features = 4 
layers = [200, 100, 5] #Number of nodes in each layer.

learning_rate = 0.001
training_steps = 2500 # 100 Epochs
batch_size = 400

###########################################################################################


def training_data(x_train, y_train, batch_size = 250, shuffles = 10000): #Shuffle and divide data up into batches.
    
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))  
    train_data = train_data.repeat().shuffle(shuffles).batch(batch_size)
    
    return train_data

def init_weights(features, hiddens): #Initialize X number of Tensors for the weights and biases. 
    
    weights = {}
    biases = {}
    random_normal = tf.initializers.RandomNormal()
    
    for idx, neurons in enumerate(hiddens):
        if idx == 0:
            weights[idx] = tf.Variable(random_normal([features, neurons]))
            biases[idx] = tf.Variable(tf.zeros([neurons]))
        else:
            weights[idx] = tf.Variable(random_normal([hiddens[idx-1], neurons]))
            biases[idx] = tf.Variable(tf.zeros([neurons]))
            
    return (weights, biases)

def model(images): #Formula for the fucntion of the neurons.
    
    for idx in range(len(weights)):
        images = tf.add(tf.matmul(images, weights[idx]), biases[idx])
        if idx != len(weights)-1:
            images = tf.nn.relu(images) #This runs on the output layer to return a probability.
            
    return tf.nn.softmax(images)

def loss_function(predictions, true):  #Multi-class Cross-entropy loss function.
    
    true = tf.one_hot(true, depth = num_classes)
    predictions = tf.clip_by_value(predictions, 1e-9, 1.) 
    
    return tf.reduce_mean(-tf.reduce_sum(true * tf.math.log(predictions)))

def optimization(images, labels): #Optimization of the weights and baises.
    
    with tf.GradientTape() as tape: #GradientTape keeps a record of previous gradients in order to calculate the new ones.
        predictions = model(images)
        loss = loss_function(predictions, labels)
        
    trainable_variables = list(weights.values()) + list(biases.values())
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

train_data = training_data(x_train, y_train, batch_size)
optimizer = tf.optimizers.Adam(learning_rate)
weights, biases = init_weights(num_features, layers)

for step, (x_batch, y_batch) in enumerate(train_data.take(training_steps), 1): #Run the functions to train the model.
    
    optimization(x_batch, y_batch)
    prediction = model(x_batch)