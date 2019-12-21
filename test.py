#-*-coding:utf-8 -*-
import sys
import numpy as np

import knn
from numpy import *
dataSet,labels = knn.createDataSet()

test_filepath='./test_data.txt'  # 数据文件路径
test_data=np.loadtxt(test_filepath,dtype=float,usecols=(1,2,3,4), delimiter='\t')

test_data_vary=np.loadtxt(test_filepath,dtype=str,usecols=(5,), delimiter='\t')
#print(test_data_vary)

test_group = array(test_data)
test_dataSize = test_group.shape[0]
#print(test_dataSize)

result=[]
K = 10
for i in range(test_dataSize):
    #print(test_group[i])
    input = array(test_group[i])
    output = knn.classify(input,dataSet,labels,K)
    result.append(output)
    print("测试样本为:",input,"分类结果为：",output)

sum = 0
for i in range(test_dataSize):
    if test_data_vary[i]==result[i]:
        sum = sum+1

print("测试准确率为:",sum/test_dataSize)



