import requests as rq
from constants import Constants
import time

totimgs = Constants().getTot()
url = "http://222.194.10.249/inc/validatecode.asp"
for i in range(1, totimgs + 1):
  res = rq.get(url)
  with open('./origin_images/{}.jpg'.format(i) ,'wb') as fb:
    fb.write(res.content)
  time.sleep(1) # 请求过快会导致多次请求到同一张图片
print("Success")