import numpy
import tensorflow
import csv
import cv2
import pandas
import numpy as np
import random

def load_data(file):
  tot_line = -1
  for index, line in enumerate(open(file, 'r')):
    tot_line += 1
  f = open(file)
  reader = csv.DictReader(f)
  labels = {}
  cnt = 0
  train_sz = tot_line // 5 * 4
  test_sz = tot_line - train_sz
  train_x = np.zeros([train_sz, 28*28])
  train_y = np.zeros([train_sz], dtype="int32")
  test_x = np.zeros([test_sz, 28*28])
  test_y = np.zeros([test_sz], dtype="int32")
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
    img = list(np.array(cv2.imread(url, cv2.IMREAD_GRAYSCALE)).ravel())
    if label not in labels:
      labels[label] = cnt
      cnt += 1
    if i < test_sz:
      test_x[i] = img
      test_y[i] = labels[label]
    else:
      train_x[i - test_sz] = img
      train_y[i - test_sz] = labels[label]
    tot += 1
    if tot % 5000 == 0:
      print(tot)

  return (train_x, train_y), (test_x, test_y), labels

if __name__ == "__main__":
  load_cifar10("D:/darkd/cifar10_train.csv")












