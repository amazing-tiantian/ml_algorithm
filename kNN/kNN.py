import datetime
from numpy import tile
import operator
import numpy as np
import matplotlib.pyplot as plt
from numpy import shape
from os import listdir

class performanceTest():
    def __init__(self):
        self.temp = 0
        self.end = 0

    def mark(self):
        if not self.end:
            self.end = datetime.datetime.now()
        else:
            self.temp = self.end
            self.end = datetime.datetime.now()

    def delta(self):
        if not self.temp:
            print 'ERROR: not enough mark points'
        else:
            delta = self.end - self.temp
            print 'RESULT: time elapsed %s' % delta



class kNN():

    def fit_and_predict(self, input, k, dataSet, label):
        dataSet_size = dataSet.shape[0]
        diff = tile(input, (dataSet_size, 1)) - dataSet
        diff_pw = diff ** 2
        euclid_dist = diff_pw.sum(axis=1)
        euclid_dist_i = euclid_dist.argsort()
        label_count = {}
        for i in range(k):
            targetLabel = label[euclid_dist_i[i]]
            label_count[targetLabel] = label_count.get(targetLabel, 0) + 1
        sorted_label_count = sorted(label_count.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sorted_label_count[0][0]

    def file2matrix(self, filename):
        f = open(filename)
        lines = f.readlines()
        f_len = len(lines)
        returnMat = np.zeros((f_len, 3))
        labelVector = []
        i = 0
        for line in lines:
            line = line.strip()
            line_splitted = line.split('\t')
            returnMat[i, :] = line_splitted[0: 3]
            labelVector.append(line_splitted[-1])
            i += 1
        return returnMat, labelVector

    def str2digi(self, labels):
        mapping = {'didntLike': 1, 'smallDoses': 2, 'largeDoses': 3}
        returnLabel = []
        for label in labels:
            returnLabel.append(int(mapping.get(label)))
        return returnLabel

    def normalize(self, dataSet):
        maxVals = dataSet.max(axis=0)
        minVals = dataSet.min(axis=0)
        span = maxVals - minVals
        normDataSet = np.zeros(shape(dataSet))
        lines = dataSet.shape[0]
        normDataSet = dataSet - np.tile(minVals,(lines, 1))
        normDataSet = normDataSet / np.tile(span, (lines, 1))
        return normDataSet, span, minVals

    def datingClassTest(self):
        trRatio = 0.10
        datingDataMat, datingLabels = self.file2matrix('data/datingTestSet2.txt')
        normMat, span, minVals = clf.normalize(datingDataMat)
        lines = normMat.shape[0]
        numTestVectors = int(lines*trRatio)
        errorCount = 0.0
        for i in range(numTestVectors):
            classifierResult = clf.fit_and_predict(normMat[i, :], 3,
                                                   normMat[numTestVectors:lines, :],
                                                   datingLabels[numTestVectors:lines])
            print "the classifier came back with: %d, the real answer is: %d" \
                  % (int(classifierResult), int(datingLabels[i]))
            if (classifierResult != datingLabels[i]):
                errorCount += 1.0
        print "the total error rate is: %f" % (errorCount/float(numTestVectors))

    def predictPerson(self):
        resultList = ['not at all', 'in small doses', 'in large doses']
        percentTime = float(raw_input('percentage of time spent playing video games?'))
        flyingMiles = float(raw_input('frequent flier miles per year?'))
        iceCreamEat = float(raw_input('liters of ice cream consumed per year?'))

        datingDataMat, datingLabels = self.file2matrix('data/datingTestSet2.txt')

        inVector = [flyingMiles, percentTime, iceCreamEat]
        normMat, span, minVals = self.normalize(datingDataMat)

        result = int(self.fit_and_predict(inVector, 3, normMat, datingLabels))
        result_tag = resultList[result - 1]
        print 'you like this person in: ' + result_tag

    def img2vector(self, filename):
        returnVector = np.zeros((1, 1024))
        f = open(filename)
        for i in range(32):
            line = f.readline()
            for j in range(32):
                returnVector[0, i*32+j] = int(line[j])
        return returnVector

    def handwritingClassTest(self):
        hwLabels = []
        trainingFileList = listdir('kNN/data/trainingDigits')
        lines = len(trainingFileList)
        trainingMat = np.zeros((lines, 1024))
        for i in range(lines):
            filename = 'kNN/data/trainingDigits/' + trainingFileList[i]
            hwLabel = int(trainingFileList[i].split('.')[0].split('_')[0])
            hwLabels.append(hwLabel)
            trainingMat[i, :] = self.img2vector(filename)

        testFileList = listdir('kNN/data/testDigits')
        errorCount = 0.0
        lines_test = len(testFileList)
        for i in range(lines_test):
            filename = 'kNN/data/testDigits/' + testFileList[i]
            label = int(testFileList[i].split('.')[0].split('_')[0])
            inVector = self.img2vector(filename)
            result = self.fit_and_predict(inVector, 3, trainingMat, hwLabels)
            print 'reading file "' + filename + '"'
            print 'the predict result is %d, the real answer is %d.' % (result, label)
            if result != label:
                errorCount += 1.0
            print '\n'

        errorRate = errorCount / float(lines_test)
        print 'number of errors is %d' % int(errorCount)
        print 'error rate is %f' % errorRate

if __name__ == '__main__':
    clf = kNN()
    p = performanceTest()
    #
    # group = np.array([[1, 1.1], [1, 1], [0, 0], [0, 0.1]])
    # labels =['A', 'A', 'B', 'B']
    #
    # print clf.fit_and_predict([0, 0], 3, group, labels)

    # mat, labels = clf.file2matrix('data/datingTestSet.txt')
    # labels = clf.str2digi(labels) #convert string labels to digital labels
    #
    # #split data
    # type1_x = []
    # type1_y = []
    # type2_x = []
    # type2_y = []
    # type3_x = []
    # type3_y = []
    #
    # for i in range(len(labels)):
    #     if labels[i] == 1: #dontLike
    #         type1_x.append(mat[i][0])
    #         type1_y.append(mat[i][1])
    #
    # for i in range(len(labels)):
    #     if labels[i] == 2:  # smallDoses
    #         type2_x.append(mat[i][0])
    #         type2_y.append(mat[i][1])
    #
    # for i in range(len(labels)):
    #     if labels[i] == 3:  # largeDoses
    #         type3_x.append(mat[i][0])
    #         type3_y.append(mat[i][1])
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111) # 335 means 3 rows and 3 columns and take the 5th piece, so does 111
    # # ax.scatter(mat[:, 0], mat[:, 1], 25.0*np.array(labels), 25.0*np.array(labels))
    # type1 = ax.scatter(type1_x, type1_y, c='red')
    # type2 = ax.scatter(type2_x, type2_y, c='green')
    # type3 = ax.scatter(type3_x, type3_y, c='blue')
    # plt.xlabel('flying miles per year')
    # plt.ylabel('time playing video games in percentage')
    # ax.legend((type1, type2, type3), ('didntLike', 'smallDoses', 'largeDoses'), loc=2) #loc parameter represents where the legend locates, default by top right
    #
    # # plt.show()
    # dataSet, span, minVals = clf.normalize(mat)
    #
    #
    # clf.datingClassTest()

    # percentTime = float(raw_input('percentage os time spent playing video games?'))
    # flyingMiles = float(raw_input('frequent flier miles per year?'))
    # iceCreamEat = float(raw_input('liters of ice cream consumed per year?'))
    #
    # inVector = [flyingMiles, percentTime, iceCreamEat]
    #
    # datingDataSet, datingLabels = clf.file2matrix('data/datingTestSet2.txt')
    #
    # # normMat, span, minVals = clf.normalize(datingDataSet)
    #
    # result = clf.fit_and_predict(inVector, 3, datingDataSet, datingLabels)
    #
    # print result

    p.mark()
    clf.handwritingClassTest()
    p.mark()
    p.delta()