#coding:utf-8
import numpy as np
from numpy import *
import operator

##给出训练数据以及对应的类别
def createDataSet():
    filepath = './train_data.txt' # 数据文件路径
    group=np.loadtxt(filepath ,dtype=float,usecols=(1,2,3,4),  delimiter='\t')

    label2=np.loadtxt('./train_data.txt' ,dtype=str, usecols=(5,), delimiter='\t')       
    labels = label2 
    
    return group,labels

###通过KNN进行分类
def classify(input,dataSet,label,k):
    dataSize = dataSet.shape[0]
    #print(dataSize)  ##135
    
    ####计算欧式距离
    diff = tile(input,(dataSize,1)) - dataSet
    sqdiff = diff ** 2
    squareDist = sum(sqdiff,axis = 1)###行向量分别相加，从而得到新的一个行向量
    dist = squareDist ** 0.5
    
    
    ##对距离进行排序
    sortedDistIndex = argsort(dist)##argsort()根据元素的值从小到大对元素进行排序，返回下标

    #print(sortedDistIndex)

    classCount={}
    for i in range(k):
        voteLabel = label[sortedDistIndex[i]]
        
        ###对选取的K个样本所属的类别个数进行统计
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    ###选取出现的类别次数最多的类别
    maxCount = 0
    for key,value in classCount.items():
        if value > maxCount:
            maxCount = value
            classes = key

    return classes
