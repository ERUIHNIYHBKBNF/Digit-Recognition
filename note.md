

~~学校网站里有个强度极低的验证码真是太方便了，这一定是学校故意留给咱练手的~~

## 基础操作

### 图片读取与保存

```python
#读取
filepath = "图片路径"
im = cv2.imread(filepath)
#保存
def saveimg(img, name):
  dirpath = "目录路径"
  filepath = os.path.join(dirpath, name) #拼接目录路径和文件名
  cv2.imwrite(filepath, img)
```

### 查看图片的方式

```python
#传入一个opencv的图片对象
def showimg(img):
  cv2.namedWindow("Image", 0) #建立窗口，第二个参数0表示自适应图片大小
  cv2.imshow("Image", img) #图片展示到窗口
  cv2.waitKey(0) #保持窗口显示直到有按键按下
```

用于测试的极弱验证码（再识别不出来就铁人工智障了）：

<img src="https://cdn.jsdelivr.net/gh/ERUIHNIYHBKBNF/picapica@main/ml-for-annual-project/2021092801.508hy4wuhwo0.png" width="300px">

## 对验证码的处理

这个大概是前半段参考的原文链接 [用Python识别验证码](https://zhuanlan.zhihu.com/p/43092916)

### 处理色彩

```python
#色彩处理部分
#灰度化
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#二值化（这里参数也许需要调整一下）
ret, im_inv = cv2.threshold(im_gray, 127, 255, cv2.THRESH_BINARY_INV)
#应用高斯模糊对图片进行降噪,高斯模糊的本质是用高斯核和图像做卷积（不懂，反正是抄的，能用就行qwq）
kernel = 1 / 16 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
im_blur = cv2.filter2D(im_inv, -1, kernel)
#降噪后再二值化处理
ret, im_res = cv2.threshold(im_blur, 127, 255, cv2.THRESH_BINARY)
```

### 获取矩形轮廓

提取文字轮廓：

```python
#返回一个列表，每个元素对应一个轮廓的所有坐标点集
contours, hierarchy = cv2.findContours(im_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

两个返回值分别是：

`contours`：一个list，每个元素是每个单独文字的轮廓（ndarray）

`hierarchy`:  `contours[i]`对应`hierarchy[i][0:4]` 四个元素，表示其后/前/父/子轮廓的索引。

\*将边框绘制到图像上：

```python
def showImgWithCons(img, cons):
  #绘制轮廓，参数：图片，轮廓，第几个（-1全部），颜色，粗细
  cv2.drawContours(img, cons, -1, (0, 0, 255), 1)
  showimg(img)
```

获取矩形轮廓：

```python
#列表中每个元素对应一个轮廓的[x, y, w, h]（左上坐标和宽度高度）
boundings = [cv2.boundingRect(cnt) for cnt in contours]
```

\*将矩形轮廓绘制到图像上：

```python
def showImgWithRect(img, boundings):
  for rect in boundings:
    [x, y, w, h] = rect
    #绘制矩形轮廓，参数：图片，左上点，右下点，颜色，粗细
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
  showimg(img)
```

### 图片大小处理

正常操作下来可以顺利拿到矩形轮廓并绘制，然而整成了这样：

<img src="https://cdn.jsdelivr.net/gh/ERUIHNIYHBKBNF/picapica@main/ml-for-annual-project/2021092802.14noltd2fr7g.png" width="300px">

看一眼二值化灰度化处理的结果，是这样子：

<img src="https://cdn.jsdelivr.net/gh/ERUIHNIYHBKBNF/picapica@main/ml-for-annual-project/2021092803.2dz19imblqqs.png" width="300px">

问了下实验室的前辈，大概是因为图片尺寸太小，这样处理：

```python
#放大图片方便处理,长和宽分别放大十倍
im = cv2.resize(im, None, fx = 10, fy = 10, interpolation = cv2.INTER_LINEAR)
#顺带调整了一些二值化的参数
ret, im_inv = cv2.threshold(im_gray, 180, 255, cv2.THRESH_BINARY_INV)
```

看起了就可以了的样子：

<img src="https://cdn.jsdelivr.net/gh/ERUIHNIYHBKBNF/picapica@main/ml-for-annual-project/2021092804.6hbe0s4bl300.png" width="300px">

找出来的最小矩形轮廓也不令人意外：

<img src="https://cdn.jsdelivr.net/gh/ERUIHNIYHBKBNF/picapica@main/ml-for-annual-project/2021092805.5jq1yr1m0gg0.png" width="300px">

### 切割轮廓并保存

> 轮廓的切割主要是通过数组切片实现的。

搜了半天没想到咋整。。突然想起这图片本身就是用数组按顺序存的像素值，直接切数组就好了唔qwq

```python
for rect in boundings:
  [x, y, w, h] = rect
  roi = im_res[y : y + h + 1, x : x + w + 1] #直接切割原图片对应数组即可
  roi = cv2.resize(roi, (30, 30)) #统一调整为30*30大小编于后期处理
  saveimg(roi, "dig" + str(int(time.time() * 1e6)) + ".jpg") #通过时间戳命名保存（防止重名）
```

看起来是这个样子的：

![](https://cdn.jsdelivr.net/gh/ERUIHNIYHBKBNF/picapica@main/ml-for-annual-project/2021092902.png)

### 功能封装

对于图片切割处理的部分，全部封装成一个函数：

传入一张验证码图片，返回一个数组，数组中的每个元素对应已经处理好的（黑白8\*8）每个数字：

```python
def divide(img):
  im = cv2.resize(im, None, fx = 10, fy = 10, interpolation = cv2.INTER_LINEAR)
  
  im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
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
```

## 收集训练数据

### 利用爬虫收集验证码图片

学校内网里的一个会随机返回~~（强度极低的）~~验证码的接口。

```python
import requests as rq
from constants import Constants
import time

totimgs = Constants().getTot()
url = "http://222.194.10.249/inc/validatecode.asp" #这个ip只支持校内访问qwq
for i in range(1, totimgs + 1):
  res = rq.get(url)
  with open('./origin_images/{}.jpg'.format(i) ,'wb') as fb:
    fb.write(res.content)
  time.sleep(1) # 请求过快会导致多次请求到同一张图片
```

顺带一提这里有个常量：（为了增加代码量显得做了很多工作（逃

constants.py:

```python
class Constants:
  __totimgs = 20 # 总共几张验证码
  def getTot(self):
    return self.__totimgs
```

### 利用OpenCV切割数字并保存

这里仍然没什么意外的地方。

```python
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
```

## 模型训练

### 手动标注训练数据

切出来的图片是这样的，像这样标记，改一下文件名就可以啦。

<img src="https://cdn.jsdelivr.net/gh/ERUIHNIYHBKBNF/picapica@main/ml-for-annual-project/2021092901.png" width="750px">

### 训练k-NN分类器

跟中期代码一样： [摸一摸k-NN算法](https://eruihniyhbkbnf.github.io/blog/2021/02/16/kNN%E7%AE%97%E6%B3%95/)

本以为ctrlCV直接完成工作自信满满结果被奇奇怪怪的异常卡了好几天QAQ~~（其实是遇到异常之后鸽了好几天）~~

#### 问题描述

先放一段一开始写的代码：

```python
totimgs =  Constants().getTot()
dir_path = "./train_data"
train_data = []
train_label = []
for i in range(0, 10):
  digit = cv2.imread(dir_path + str(i) + ".jpg")
  train_data.append(digit)
  train_label.append(i)
train_data = np.array(train_data).astype(np.float32)
train_label = np.array(train_label).astype(np.float32)

knn = cv2.ml.KNearest_create()
knn.train(train_data, cv2.ml.ROW_SAMPLE, train_label)
```

先是不认识的蜜汁报错：

<img src="https://cdn.jsdelivr.net/gh/ERUIHNIYHBKBNF/picapica@main/ml-for-annual-project/2021100401.png" width="750px">

搜一波全是OpenCV调用摄像头的错误qwq（咱哪里调过摄像头啊呜呜呜QAQ

#### 解决过程

总之输出一下train_data看看：

```python
with open("test.txt","w") as f:
  f.write(str(train_data[0]))
```

test.txt:

```text
[array([[[  0,   0,   0],
        [255, 255, 255],
        [254, 254, 254],
        [255, 255, 255],
        [255, 255, 255],
        [253, 253, 253],
        [255, 255, 255],
        [  0,   0,   0]],

       [[254, 254, 254],
        [174, 174, 174],
        [  2,   2,   2],
        [  1,   1,   1],
        ......
```

对比之前ac的代码可以发现，原来用范围0~1的float32表示颜色（或者该叫黑白程度？），而这里通过imread读到的却是rgb颜色值

对比：

```python
array([[ 0.,  0.,  5., 13.,  9.,  1.,  0.,  0.],
       [ 0.,  0., 13., 15., 10., 15.,  5.,  0.],
       [ 0.,  3., 15.,  2.,  0., 11.,  8.,  0.],
       [ 0.,  4., 12.,  0.,  0.,  8.,  8.,  0.],
       [ 0.,  5.,  8.,  0.,  0.,  9.,  8.,  0.],
       [ 0.,  4., 11.,  0.,  1., 12.,  7.,  0.],
       [ 0.,  2., 14.,  5., 10., 12.,  0.,  0.],
       [ 0.,  0.,  6., 13., 10.,  0.,  0.,  0.]])
```

（然后搜了一波颜色的表示方法还有rgb如何转float32之类的还是没看懂QAQ~~（其实是懒得看）~~

后来突然悟了。。**knn本身就是算个距离，这边rgb又都一样（因为灰度二值处理过），取rgb任意一个值转成float32用就行**，也没必要弄成0~1之间的值啦。

utils.py添加函数：

传入一个二维数组（imread）（其实是三维, $img[i][j]$ 是一个长为3的列表，表示的是图片第 i 行 j 列像素点的rgb）转换成np的一维向量形式，详见注释：

```python
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
```

#### 训练模型

```python
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
  # 转换成ndarray表示的
  digit = rgbToFloat32(img)
  train_data.append(digit)
  train_label.append(i)

train_data = np.array(train_data, dtype='float32')
train_label = np.array(train_label, dtype='float32')

knn = cv2.ml.KNearest_create()
knn.train(train_data, cv2.ml.ROW_SAMPLE, train_label)
# 拿自己测一下看看应该没什么问题的样子
_, res, _, _ = knn.findNearest(train_data, k=1)
print(mts.accuracy_score(res, train_label))
# 1.0
```

### 结果测试与性能评估

顺带在knn.py里塞一个函数：

```python
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
```

再用img_spider.py爬一些验证码来存个别的文件夹里：

```python
with open('./test_images/{}.jpg'.format(i) ,'wb') as fb:
```

test.py:

```python
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
```

<img src="https://cdn.jsdelivr.net/gh/ERUIHNIYHBKBNF/picapica@main/ml-for-annual-project/2021100404.png" height="350px">

爽啊(´▽｀)ノ♪

## 应用到网络爬虫

爬取一张验证码并获取值：~~（从哪里爬起来的就从哪里继续爬）~~

```python
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
# showimg(img)
print(value)
# 记得删除文件qwq
os.remove(fileName)
```

然后登录提交表单带上验证码就可以拿到饼干了：

```python
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
print(cookies) # 这里好像还有点问题, 不过不影响大局就是了反正装个样子qwq（逃
```

忙死啦忙死啦先跑路了(´д⊂)
