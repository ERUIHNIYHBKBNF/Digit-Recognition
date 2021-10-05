import requests as rq
import cv2
import time
import os
from picdiv import divide
from knn import getVeryValue
from utils import showimg

url = 'http://222.194.10.249/inc/validatecode.asp'
res = rq.get(url)
# 文件名加个时间戳
fileName = str(int(time.time())) + '.jpg'
# 由于不会在内存中直接转换二进制到rgb就只能存了再读了qwq
with open(fileName, 'wb') as f:
  f.write(res.content)
img = cv2.imread(fileName)
digits = divide(img)
res = getVeryValue(digits)
value = 0
for i in res[::-1]:
  value = value * 10 + int(i)
showimg(img)
print(value)
os.remove(fileName)

url = 'http://222.194.10.249/checklogin.asp'
headers = {
  'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
  'Accept-Encoding': 'gzip, deflate',
  'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
  'Cache-Control': 'no-cache',
  'Connection': 'keep-alive',
  'Content-Length': '69',
  'Content-Type': 'application/x-www-form-urlencoded',
  'Cookie': 'ASPSESSIONIDQABSAQDS=EBCFBPHBMKFIIBBNLHJMCHKJ; XINHAI_Admin_Id=; XINHAI_Admin_Password=; XINHAI_Admin_Right=; XINHAI%5FAdmin=; XINHAI_Student_Id=; XINHAI_Student_Password=; XINHAI=; XINHAI%5FStudent=; XINHAI_Message=',
  'Host': '222.194.10.249',
  'Origin': 'http://222.194.10.249',
  'Pragma': 'no-cache',
  'Referer': 'http://222.194.10.249/index.asp',
  'Upgrade-Insecure-Requests': '1',
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36'
}
data = {
  'usertype': 'Student',
  'username': '用户名',
  'password': '密码',
  'validate': value
}
res = rq.post(url = url, headers = headers, data = data)
cookies = rq.utils.dict_from_cookiejar(res.cookies)
print(cookies)
