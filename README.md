# Digit-Recognition

低配版的python手写数字识别\&验证码识别。

大一年度项目 手写数字识别，啥都不懂的情况下自信选题，黑历史+1。

不要看了快走开啦！(＞＜)

## 中期内容

用OpenCV自带的 k-NN 实现了手写数字识别，文件 `digit_recognition.py`

报告在这里：https://eruihniyhbkbnf.github.io/blog/2021/02/16/kNN%E7%AE%97%E6%B3%95/

## 结题内容

~~这不是中期就把项目做完了吗，怎么还要应用QAQ~~

于是要做一下验证码识别~~（做这个为什么不直接调现成的api？~~

~~总之为了显得工作量很大就在：收集验证码，训练和测试数据准备，应用到网络爬虫。这几个方面瞎写了些没用的代码~~

碰巧学校有个年代久远的~~辣鸡~~网站和强度极低的验证码可以拿来练手，感谢学校提供的学习资源！（逃）

## 项目文件介绍

文件夹：

`origin_images`: 存放爬到的一些验证码，用于生成训练数据。

`train_data`: 把验证码进行灰度化二值化处理后，切割成数字，数字保存到这里。挑取`0~9`总共十个数字做训练集即可。（没错就只需要十个数字，因为这验证码里面每个数字长得都一样enmm）

示例：<img src="https://cdn.jsdelivr.net/gh/ERUIHNIYHBKBNF/picapica@main/ml-for-annual-project/2021092801.508hy4wuhwo0.png" width="300px">

`test_data`: 爬取的另一些验证码图片，用于测试效果。
文件：

`digit_recognition.py`: 中期的手写数字识别实现~~（通过调用各种库）~~，与本项目无关。

`constants.py`: 常量，~~好像只有一个变量但为了显得工作量很大....~~。

`utils.py`: 一些工具，包括保存图片、查看图片、rgb转float32之类的。

`img_spider.py`: 验证码爬虫，从学校网站里爬取~~低质量~~验证码。

`picdiv.py`: 切割图片，传入一个验证码图片，返回四个黑白数字图片。

`train_data_maker.py`: 把刚刚爬到的验证码做成黑白单个数字并保存。

`knn.py`: 训练knn模型用于数字识别，~~然而只需要十张训练数据。~~

`test.py`: 用一些新的验证码进行测试。

`login_spider.py`: 模拟登录过程拿到饼干。（好像还有什么问题，不过就表示一下这个意思啦QwQ）

`node.md`: 过程记录。

## 食用方法

1. 运行 `img_spider.py` 爬取一些验证码图片到 `origin_images` 文件夹内。
2. 运行 `train_data_maker.py` 存一堆数字到 `train_data` 文件夹内。
3. 进入 `train_data` 文件夹，找出 `0~9` 各一个数字并手动命名为 `*.jpg`，对应数字值。
4. 把 `img_spider.py` 的文件夹路径改为 `test_data` ，然后运行`test.py` ，或者直接运行 `login_spider.py` 进行测试。

~~下次项目再选机器学习咱就是傻宝QAQ~~

