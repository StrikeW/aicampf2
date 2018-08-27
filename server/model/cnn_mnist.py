""" Convolutional Neural Network.

Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

This example is using TensorFlow layers API, see 'convolutional_network_raw' 
example for a raw implementation with variables.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import division, print_function, absolute_import

import numpy as np
import sys
import os
from server.model import datasets
import tensorflow as tf
import pandas as pd
from tensorflow.contrib import predictor
from server import config
import json


# Training Parameters
learning_rate = 0.001
num_steps = 100
batch_size = 128

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.25 # Dropout, probability to drop a unit

#set parameter
def set_parameter(dic):
    # dic = json.loads(conf)[0]
    global learning_rate, num_steps, batch_size
    learning_rate = dic['learning_rate'];
    num_steps = dic['num_steps']
    batch_size = dic['batch_size']

# Create the neural network
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict['images']

        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)

    return out


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, num_classes, dropout, reuse=False,
                            is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True,
                           is_training=False)

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)


    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        print('PREDICT mode')
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs

def evaluate(model, x_test, y_test):
    # Evaluate the Model
    # Define the input function for evaluating
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': x_test}, y=y_test,
        batch_size=batch_size, shuffle=False)
    # Use the Estimator 'evaluate' method
    e = model.evaluate(input_fn)

    print("Evaluate Accuracy:", e['accuracy'])

    return e['accuracy'], e['loss']

# 使用predictor.from_saved_model()加载导出的模型，用来预测！
def predict(export_dir, x_test):
    print('Predict mode')
    predict_fn = predictor.from_saved_model(export_dir)
    predictions = predict_fn( {"images": x_test} )
    df = pd.DataFrame(predictions)
    # df['original_labels'] = y_test
    print(df.head())
    # total = len(predictions['output'])
    # count = 0
    # for i in range(total):
        # if predictions['output'][i] == y_test[i]:
            # count += 1

    # accuracy = count/total
    # print("Predict Accuracy:", accuracy)
    return "Predict Accuracy:" + str(accuracy)

def train():
    (x_train, y_train), (x_test, y_test) = datasets.load_mnist()
    x_train = x_train.astype(np.float32).reshape(x_train.shape[0], 784)
    y_train = y_train.astype(np.float32)

    x_test = x_test.astype(np.float32).reshape(x_test.shape[0], 784)
    y_test = y_test.astype(np.float32)

    print('Train mode')
    # tf.logging.set_verbosity(tf.logging.INFO)
    # Build the Estimator
    model = tf.estimator.Estimator(model_fn)

    # Define the input function for training
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': x_train}, y=y_train,
        batch_size=batch_size, num_epochs=None, shuffle=True)

    # Train the Model
    model.train(input_fn, steps=num_steps)
    acc, loss = evaluate(model, x_test, y_test)

    feat_spec = {"images": tf.placeholder("float", name="images", shape=[None, x_train.shape[1]])}
    # print(feat_spec)

    # Export model
    receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feat_spec)
    saved_estimator_path = model.export_savedmodel('saved_model', receiver_fn).decode("utf-8")
    print('model is saved to [%s]' % saved_estimator_path)

    model_info = {}
    model_info['acc'] = acc
    model_info['save_path'] = os.path.join(config.project_root, saved_estimator_path)
    model_info['loss'] = loss
    return model_info

if __name__ == "__main__":
    train()
    if len(sys.argv) <= 1:
        print('lack of arguments!')
        exit(-1)

    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'predict':
        (x_train, y_train), (x_test, y_test) = datasets.load_mnist()
        predict(sys.argv[2], x_test)

