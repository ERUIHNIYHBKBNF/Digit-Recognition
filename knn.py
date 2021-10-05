import numpy as np
import cv2
from constants import Constants
from utils import rgbToFloat32
from sklearn import metrics as mts

totimgs =  Constants().getTot()
dir_path = "./train_data/"
train_data = []
train_label = []

for i in range(0, 10):
  # imread读入的是rgb值，返回一个三维的ndarray,[行[列[RGB]]]
  img = cv2.imread(dir_path + str(i) + ".jpg")
  digit = rgbToFloat32(img)
  train_data.append(digit)
  train_label.append(i)

train_data = np.array(train_data, dtype='float32')
train_label = np.array(train_label, dtype='float32')

knn = cv2.ml.KNearest_create()
knn.train(train_data, cv2.ml.ROW_SAMPLE, train_label)
_, res, _, _ = knn.findNearest(train_data, k=1)

def getVeryValue(digits):
  test_data = []
  for i in range(0, len(digits)):
    npDigit = []
    for j in range(0, len(digits[i])):
      for k in range(0, len(digits[i][j])):
        npDigit.append(digits[i][j][k])
    test_data.append(np.array(npDigit, dtype='float32'))
  test_data = np.array(test_data, dtype='float32')
  _, res, _, _ = knn.findNearest(test_data, k=1)
  return res