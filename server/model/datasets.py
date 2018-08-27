import numpy
import tensorflow
import csv
import cv2
import numpy as np

def load_mnist():
  return tensorflow.keras.datasets.mnist.load_data()

def load(file):
  f = open(file)
  reader = csv.DictReader(f)
  labels = {}
  cnt = 0
  x = []
  y = []
  for row in reader:
    if cnt == 0:
      cnt += 1
      continue
    #print(row)
    url, label = row['url'], row['label']
    img = np.array(cv2.imread(url)).ravel()
    x.append(img)

    if label not in labels:
      labels[label] = cnt
      cnt += 1

    y.append(labels[label])
  return (x, y), labels

def load_cifar10(file):
  (train_x, train_y),labels = load(file + "train.csv")
  (test_x, test_y),_ = load(file + "test.csv")
  return (train_x, train_y), (test_x, test_y),labels

if __name__ == "__main__":
  load_cifar10("D:/darkd/cifar10_")