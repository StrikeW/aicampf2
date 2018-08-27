import numpy
import tensorflow
import csv
import cv2
import pandas
import numpy as np

def load_mnist():
  return tensorflow.keras.datasets.mnist.load_data()

def load(file):
  tot_line = -1
  for index, line in enumerate(open(file, 'r')):
    tot_line += 1
  f = open(file)
  reader = csv.DictReader(f)
  labels = {}
  cnt = 0
  tot = -1
  x = np.zeros([tot_line, 32*32*3])
  y = np.zeros([tot_line], dtype='int32')
  for row in reader:
    if tot == -1:
      tot += 1
      continue
    #print(row)
    url, label = row['url'], row['label']
    img = list(np.array(cv2.imread(url)).ravel())
    x[tot] = img
    if label not in labels:
      labels[label] = cnt
      cnt += 1
    y[tot] = labels[label]
    tot += 1
    if tot % 5000 == 0:
      print(img)
      print(tot)

  return (x, y), labels

def load_cifar10(file):
  (train_x, train_y),labels = load(file)
  (test_x, test_y),_ = load(file)
  return (train_x, train_y), (test_x, test_y),labels

if __name__ == "__main__":
  load_cifar10("D:/darkd/cifar10_")