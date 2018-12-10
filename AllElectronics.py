#!/usr/bin/env python
#coding: UTF-8
from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import tree
from sklearn import preprocessing
from sklearn.externals.six import StringIO

# Read in the csv file and put features into list of dict and list of class label
allElectronicsData = open('AllElectronics.csv', 'rt')
reader = csv.reader(allElectronicsData)
headers = next(reader)  #读取csv文件中的第一行标题header 特征名称
print(headers)

#把所有的特征放在list中，每一行特征放在一个字典中。
#list中的每一个字典对应原始数据中的一行数据
featureList = []
labelList = []

for row in reader:
    labelList.append(row[len(row)-1])
    rowDict = {} #每一行的特征都存一个字典里面
    for i in range(1, len(row)-1):  #第一列是RID，不是特征。最后一列是类别标签。
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)
# print(labelList)
print(featureList)

# Vectorize features 矢量化特征值，把特征值转换成 0/1的形式。
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList) .toarray() #输出转化后的特征矩阵
print("dummyX: " + str(dummyX))
print(vec.get_feature_names()) #查看特征的哪个值是0，哪个是1。

print("labelList: " + str(labelList))

# vectorize class labels 矢量化类别标签
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY: " + str(dummyY))

# Using decision tree for classification
# clf = tree.DecisionTreeClassifier()
clf = tree.DecisionTreeClassifier(criterion='entropy') #使用信息熵作为划分标准，对决策树进行训练
#print(clf)
clf = clf.fit(dummyX, dummyY) #训练数据集：用训练数据拟合分类器模型
print("clf: " + str(clf))

# Visualize model 将获得的决策树写入dot文件：
with open("allElectronicInformationGainOri.dot", 'w') as f: #自动先创建一个文件，然后写入内容
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

oneRowX = dummyX[0, :]
print("oneRowX: " + str(oneRowX))

# 制造一行数据，用来测试算法
newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print("newRowX: " + str(newRowX)) # 测试数据

# 预测
predictedY = clf.predict(newRowX.reshape(1,-1))
print("predictedY: " + str(predictedY))


