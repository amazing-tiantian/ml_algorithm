import numpy as np
import datetime
import auth.kNN as kNN
from os import listdir

# def calEuclidDist(X, Y):
#     d = 0
#     for x, y in zip(X, Y):
#         d += pow((x - y), 2)
#     return d
#
# def calEuclidDist_new(X, Y):
#     diff = X - Y
#     diff_pw = diff ** 2
#     d = diff_pw.sum()
#     return d
#
#
# x = [1, 2, 3]
# y = [2, 4, 6]
#
# x = np.array(x)
# y = np.array(y)
#
# x_sort = y.sort()
# print y


# start = datetime.datetime.now()
# print calEuclidDist_new(x, y)
# end = datetime.datetime.now()
# delta = end - start
# print delta
#
# start = datetime.datetime.now()
# print calEuclidDist(x, y)
# end = datetime.datetime.now()
# delta = end - start
# print delta


# dict = {}
#
# dict['a'] = dict.get('a', 0) + 1
# print dict
# dict['a'] = dict.get('a', 0) + 1
# print dict

# dataSet = np.array([[2, 4, 7], [6, 11, 16]])
# print dataSet.max(axis=1)
# print dataSet.shape[0]
# print dataSet.shape[1]

# x = np.array([1,2])
# print np.tile(x,(2,2))

# s = "Today is a beautiful day, I wanna to go out!"
# # print s
# s_list = s.split()
# print s_list
# for line in s_list:
#     print line

# x = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 0]])
# lines = x.shape[0]
# print x[0:, :]

# def datingClassTest():
#     clf = kNN()
#     mat, labels = clf.file2matrix('/data/datingTestSet2.txt')
#     ratio = 0.10
#     normMat = clf.normalize(mat)
#     lines = normMat.shape[0]
#     vectorNum = (int)(lines * ratio)
#     error_num = 0.0
#     for i in range(vectorNum):
#         predict_result = clf.fit_and_predict(normMat[i, :],3 ,normMat[vectorNum: lines, :], labels[vectorNum: lines])
#         print 'predict_result is %d, real_result is %d' % predict_result, labels[i]
#         if predict_result != labels[i]:
#             error_num += 1.0
#     print 'error rate: %f' % (error_num/float(vectorNum))

# datingDataSet, datingDataLabels = kNN.file2matrix('data/datingTestSet2.txt')
#
# normMat, span, minVals = kNN.autoNorm(datingDataSet)
#
# percentTime = float(raw_input('percentage os time spent playing video games?'))
# flyingMiles = float(raw_input('frequent flier miles per year?'))
# iceCreamEat = float(raw_input('liters of ice cream consumed per year?'))
#
# inVector = [flyingMiles, percentTime, iceCreamEat]
#
# result = kNN.classify0(inVector, normMat, datingDataLabels, 3)
# print result

# trainingFileList = listdir('data/trainingDigits')
# filename = trainingFileList[25]
# print filename
# file = filename.split('.')
# file = file[0].split('_')[0]

# def a(menu):
#     menu[0] = 1000000
#     return menu
#
# list =[1, 2, 3, 4, 5]
#
# print list
# x = a(list)
# print x
# print list






