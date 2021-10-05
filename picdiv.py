import cv2
import numpy as np
from utils import showimg

def divide(img):
  img = cv2.resize(img, None, fx = 10, fy = 10, interpolation = cv2.INTER_LINEAR)
  
  im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ret, im_inv = cv2.threshold(im_gray, 180, 255, cv2.THRESH_BINARY_INV)
  kernel = 1 / 16 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
  im_blur = cv2.filter2D(im_inv, -1, kernel)
  ret, im_res = cv2.threshold(im_blur, 127, 255, cv2.THRESH_BINARY)

  contours, hierarchy = cv2.findContours(im_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  boundings = [cv2.boundingRect(cnt) for cnt in contours]

  digits = []
  for rect in boundings:
    [x, y, w, h] = rect
    roi = im_res[y : y + h + 1, x : x + w + 1]
    roi = cv2.resize(roi, (8, 8))
    digits.append(roi)

  return digits