from picdiv import divide
from knn import getVeryValue
from constants import Constants
import cv2

totimgs = Constants().getTot()
dir_path = "./test_images/"
for i in range(1, totimgs + 1):
  veryCode = cv2.imread(dir_path + str(i) + ".jpg")
  digits = divide(veryCode) #这里digits应该是长为4的列表，每个元素[行[列[颜色值0-255不是数组]]]
  res = getVeryValue(digits)
  value = 0
  for i in res[::-1]:
    value = value * 10 + int(i)
  print(value)
  