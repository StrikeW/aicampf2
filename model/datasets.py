import numpy
import tensorflow as tf

def load_mnist():
  return tf.keras.datasets.mnist.load_data()

def load_cifar10():
  return tf.keras.datasets.cifar10.load_data()

