# !/usr/bin/python
# coding=utf-8
#########################################
# kNN: k Nearest Neighbors

#  输入:      newInput:  (1xN)的待分类向量
#             dataSet:   (NxM)的训练数据集
#             labels:     训练数据集的类别标签向量
#             k:         近邻数

# 输出:     可能性最大的分类标签
#########################################

import numpy as np
import pandas


# KNN分类算法函数定义
def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]  # shape[0]表示行数

    # # step 1: 计算距离[
    # 假如：
    # Newinput：[1,0,2]
    # Dataset:
    # [1,0,1]
    # [2,1,3]
    # [1,0,2]
    # 计算过程即为：
    # 1、求差
    # [1,0,1]       [1,0,2]
    # [2,1,3]   --   [1,0,2]
    # [1,0,2]       [1,0,2]
    # =
    # [0,0,-1]
    # [1,1,1]
    # [0,0,-1]
    # 2、对差值平方
    # [0,0,1]
    # [1,1,1]
    # [0,0,1]
    # 3、将平方后的差值累加
    # [1]
    # [3]
    # [1]
    # 4、将上一步骤的值求开方，即得距离
    # [1]
    # [1.73]
    # [1]

    # tile(M, reps): 把M重复reps次，构造一个矩阵
    diff = np.tile(newInput, (numSamples, 1)) - dataSet  # 按元素求差值
    squaredDiff = diff ** 2  # 将差值平方
    squaredDist = np.sum(squaredDiff, axis=1)  # 按行累加
    distance = squaredDist ** 0.5  # 将差值平方和求开方，即得距离

    # # step 2: 对距离排序
    # argsort() 返回排序后的索引值
    sortedDistIndices = np.argsort(distance)
    classCount = {}  # define a dictionary (can be append element)
    for i in range(k):
        # # step 3: 选择k个最近邻
        voteLabel = labels[sortedDistIndices[i]]

        # # step 4: 计算k个最近邻中各类别出现的次数
        # when the key voteLabel is not in dictionary classCount, get() will return 0
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    # # step 5: 返回出现次数最多的类别标签
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key

    return maxIndex


if __name__ == 'main':
    # testX = np.array([12.71,2.21,2.3,23.9,97.9,1.05,0.99])
    k = 5

    DATA_PATH = './wine.csv'
    TEST_PATH = './test.csv'

    data_matrix = pandas.read_csv(DATA_PATH, header=0)
    test_matrix = pandas.read_csv(TEST_PATH, header=0)

    #  display(data_matrix)
    dataSet = np.array(
        data_matrix[["Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium", "Total Phenols", "Flavanoids"]])
    labels = list(data_matrix["type"])
    for index, row in test_matrix.iterrows():
        # testX = np.array(row[["Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium", "Total Phenols", "Flavanoids"]])
        testX = np.array([12.71, 2.21, 2.3, 23.9, 97.9, 1.05, 0.99])

        # 调用分类函数对未知数据分类
        outputLabel = kNNClassify(testX, dataSet, labels, k)
        print("Your input is:", testX, "and classified to class:", outputLabel)



 # testX = np.array([12.71,2.21,2.3,23.9,97.9,1.05,0.99])
k = 5
DATA_PATH = './wine.csv'
TEST_PATH = './test.csv'

data_matrix = pandas.read_csv(DATA_PATH, header=0)
test_matrix = pandas.read_csv(TEST_PATH, header=0)

#  display(data_matrix)
dataSet = np.array(
    data_matrix[["Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium", "Total Phenols", "Flavanoids"]])
labels = list(data_matrix["type"])
for index, row in test_matrix.iterrows():
    # testX = np.array(row[["Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium", "Total Phenols", "Flavanoids"]])
    testX = np.array([12.71, 2.21, 2.3, 23.9, 97.9, 1.05, 0.99])

    # 调用分类函数对未知数据分类
    outputLabel = kNNClassify(testX, dataSet, labels, k)
    print("Your input is:", testX, "and classified to class:", outputLabel)