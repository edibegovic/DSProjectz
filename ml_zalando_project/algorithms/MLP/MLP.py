###### Tensorflow 2 #######

from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np

#Run to upgrade Tensorflow in Google Colabs to 2.
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

#Number of nodes in each layer.
layers = [200, 100, 5] 

learning_rate = 0.001

# 100 Epochs
training_steps = 2500 
batch_size = 400

###########################################################################################

# Shuffle and divide data up into training batches.
def training_data(x_train, y_train, batch_size = 250, shuffles = 10000): 
    
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))  
    # Shuffle and batch data.
    train_data = train_data.repeat().shuffle(shuffles).batch(batch_size) 
    
    return train_data

# Initialize Tensors for the weights and biases of each layer.
def init_weights(features, hiddens): 
    
    weights = {}
    biases = {}
    # Initialize weights 
    random_normal = tf.initializers.RandomNormal() 
    
    for idx, neurons in enumerate(hiddens):
        if idx == 0:
            weights[idx] = tf.Variable(random_normal([features, neurons])) 
            biases[idx] = tf.Variable(tf.zeros([neurons]))
        else:
            weights[idx] = tf.Variable(random_normal([hiddens[idx-1], neurons]))
            biases[idx] = tf.Variable(tf.zeros([neurons]))
            
    return (weights, biases)

# Formula for the fucntion of the neurons.
def model(images): 
    
    for idx in range(len(weights)):
        images = tf.add(tf.matmul(images, weights[idx]), biases[idx])
        if idx != len(weights)-1:
            #ReLU activation fucntion
            images = tf.nn.relu(images) 
            
    # This runs on the output layer to return a probability.
    return tf.nn.softmax(images)    

# Multi-class Cross-entropy loss function.
def loss_function(predictions, true):  
    
    true = tf.one_hot(true, depth = num_classes)
    #Clip values to an interval to avoid log problems.
    predictions = tf.clip_by_value(predictions, 1e-9, 1.) 
    
    return tf.reduce_mean(-tf.reduce_sum(true * tf.math.log(predictions)))

# Optimization of the weights and baises.
def optimization(images, labels): 
    
    # GradientTape keeps a record of previous gradients in order to calculate the new ones.
    with tf.GradientTape() as tape: 
        predictions = model(images)
        loss = loss_function(predictions, labels)
        
    trainable_variables = list(weights.values()) + list(biases.values())
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

train_data = training_data(x_train, y_train, batch_size)
optimizer = tf.optimizers.Adam(learning_rate)
weights, biases = init_weights(num_features, layers)

# Run the functions to train the model
for step, (x_batch, y_batch) in enumerate(train_data.take(training_steps), 1): 
    
    optimization(x_batch, y_batch)
    prediction = model(x_batch)

#Reference to:
#https://github.com/aymericdamien/TensorFlow-Examples/tree/master/tensorflow_v2
