import numpy as np
from cv2 import cv2
from sklearn import datasets as ds
from sklearn import metrics as mts
#数据准备
digits = ds.load_digits()
knn = cv2.ml.KNearest_create()
train_data = digits.data[:1000].astype(np.float32) #前 999 个用来训练，转为 float32 便于读取
train_labels = digits.target[:1000]
test_data = digits.data[1000:].astype(np.float32)
test_labels = digits.target[1000:]
#模型训练
knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
#测试
_, res, _, _ = knn.findNearest(test_data, k=10)
print(mts.accuracy_score(res, test_labels)) #正确率 0.9560853199498118
