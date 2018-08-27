import numpy
import tensorflow

def load_mnist():
  return tensorflow.keras.datasets.mnist.load_data()

def load_cifar10():
  return tensorflow.keras.datasets.cifar10.load_data()

