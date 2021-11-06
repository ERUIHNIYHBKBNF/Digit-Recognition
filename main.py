import time
from selenium import webdriver
import cv2
from utils import *
from picdiv import divide
from knn import getVeryValue
import pyautogui
import os
import pyperclip

from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys

browser = webdriver.Chrome()
browser.get('http://222.194.10.249/')
filepath = './valiCode.jpg'

time.sleep(2)
imgElem = browser.find_element_by_xpath('/html/body/div[1]/div[2]/div[2]/div[1]/div[2]/form/label[4]/img')
action = ActionChains(browser)
action.context_click(imgElem).perform()
time.sleep(2)
pyautogui.typewrite('v') # 保存
time.sleep(0.5)
pyautogui.typewrite(os.path.abspath(filepath))
time.sleep(2)
# pyautogui.hotkey('ctrlleft', 'v')  # 粘贴
# time.sleep(0.5)
pyautogui.press('enter')
time.sleep(2)

img = cv2.imread(filepath)
digits = divide(img)
res = getVeryValue(digits)
value = 0
for i in res[::-1]:
  value = value * 10 + int(i)
os.remove(filepath)
print(value)

username = '2201110212'
password = '123456'
browser.find_element_by_xpath('/html/body/div/div[2]/div[2]/div[1]/div[2]/form/label[2]/input').click()
pyautogui.typewrite(username)
time.sleep(0.5)
#browser.find_element_by_xpath('/html/body/div/div[2]/div[2]/div[1]/div[2]/form/label[3]/input').click()
pyautogui.press('tab')
pyautogui.typewrite(password)
time.sleep(0.5)
#browser.find_element_by_xpath('/html/body/div/div[2]/div[2]/div[1]/div[2]/form/label[4]/input').click()
pyautogui.press('tab')
pyautogui.typewrite(str(value))
time.sleep(2)

browser.find_element_by_xpath('/html/body/div/div[2]/div[2]/div[1]/div[2]/form/div/input[1]').click()

input()
