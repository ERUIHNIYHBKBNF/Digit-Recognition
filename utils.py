import cv2
import numpy as np
import os

def saveimg(img, name):
  dirpath = "./train_data"
  filepath = os.path.join(dirpath, name)
  cv2.imwrite(filepath, img)

def showimg(img):
  cv2.namedWindow("Image", 0)
  cv2.imshow("Image", img)
  cv2.waitKey(0)

def showImgWithCons(img, cons):
  cv2.drawContours(img, cons, -1, (0, 0, 255), 1)
  showimg(img)

def showImgWithRect(img, boundings):
  for rect in boundings:
    [x, y, w, h] = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
  showimg(img)

# OpenCV的knn模型要求train_data应该是一个二维数组
# 其中每个元素都是将图片展开成一维，值是float32表示的颜色值
# 构造一个digit数组，长度是img的 hegiht * width
# 由于img灰度二值处理过，rgb三个值相同取一个即可
def rgbToFloat32(img):
  data = []
  for i in range(0, len(img)):
    for j in range(0, len(img[i])):
      data.append(img[i][j][0])
  return np.array(data, dtype='float32')