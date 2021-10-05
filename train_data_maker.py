from picdiv import divide
import cv2
from constants import Constants
from utils import saveimg

totimgs = Constants().getTot()
for i in range(1, totimgs + 1):
  veryCode = cv2.imread("./origin_images/" + str(i) + '.jpg')
  digits = divide(veryCode)
  for j in range(0, len(digits)):
    saveimg(digits[j], str((i - 1) * 4 + j + 1) + '.jpg')