import numpy
import tensorflow
import csv
import cv2
import pandas
import numpy as np
import random

def load_mnist():
  return tensorflow.keras.datasets.mnist.load_data()

def load_cifar10(file):
  tot_line = -1
  for index, line in enumerate(open(file, 'r')):
    tot_line += 1
  f = open(file)
  reader = csv.DictReader(f)
  labels = {}
  cnt = 0
  train_x = np.zeros([tot_line // 5 * 4, 32*32*3])
  train_y = np.zeros([tot_line // 5 * 4], dtype="int32")
  test_x = np.zeros([tot_line // 5 * 4, 32*32*3])
  test_y = np.zeros([tot_line // 5 * 4], dtype="int32")
  id = []
  for i in range(tot_line):
    id.append(i)
  random.shuffle(id)
  tot = -1;
  for row in reader:
    if tot == -1:
      tot += 1
      continue
    i = id[tot]
    url, label = row['url'], row['label']
    img = list(np.array(cv2.imread(url)).ravel())
    if label not in labels:
      labels[label] = cnt
      cnt += 1
    if i < tot_line // 5:
      test_x[i] = img
      test_y[i] = labels[label]
    else:
      train_x[i - tot_line // 5] = img
      test_y[i - tot_line // 5] = labels[label]
    tot += 1
    if tot % 5000 == 0:
      print(tot)

  return (train_x, train_y), (test_x, test_y), labels

if __name__ == "__main__":
  load_cifar10("D:/darkd/cifar10_train.csv")












