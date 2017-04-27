from math import log
import operator
import matplotlib.pyplot as plt

class dTree():
    def __init__(self):
        self.decisionNode = dict(boxstyle='sawtooth', fc='0.8')
        self.leafNode = dict(boxstyle='round4', fc='0.8')
        self.arrow_args = dict(arrowstyle='<-')

    def calShannonEnt(self, dataSet):
        totalNum = len(dataSet)
        labelCounts = {}
        for featVec in dataSet:
            currentLabel = featVec[-1]
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        shannonEnt = 0.0
        for key in labelCounts:
            p = float(labelCounts[key]) / totalNum
            shannonEnt -= p * log(p, 2)
        return shannonEnt

    def createDataSet(self):
        dataSet = [[1, 1, 'yes'],
                   [1, 1, 'yes'],
                   [1, 0, 'no'],
                   [0, 1, 'no'],
                   [0, 1, 'no']]
        labels = ['no surfacing', 'flippers']
        return dataSet, labels

    def splitDataSet(self, dataSet, axis, value):
        retDataSet = []
        for featVec in dataSet:
            if featVec[axis] == value:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reducedFeatVec)
        return retDataSet

    def chooseBestFeatureToSplit(self, dataSet):
        numFeatures = len(dataSet[0]) - 1
        baseEntropy = self.calShannonEnt(dataSet)
        bestInfoGain = 0.0
        bestFeature = -1
        for i in range(numFeatures):
            featList = [example[i] for example in dataSet]
            uniqueVals = set(featList)
            newEntropy = 0.0
            for value in uniqueVals:
                subDataSet = self.splitDataSet(dataSet, i ,value)
                p = len(subDataSet) / float(len(dataSet))
                newEntropy += p * self.calShannonEnt(subDataSet)
            infoGain = baseEntropy - newEntropy
            if (infoGain > bestInfoGain):
                bestInfoGain = infoGain
                bestFeature = i
        return bestFeature

    def majorityCnt(self, classList):
        classCount = {}
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.iteritems(),
                                  key=operator.itemgetter(1),
                                  reverse=True)
        return sortedClassCount[0][0]

    def createTree(self, dataSet, labels):
        classList = [data[-1] for data in dataSet]
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        if len(dataSet[0]) == 1:
            return self.majorityCnt(classList)
        bestFeat = self.chooseBestFeatureToSplit(dataSet)
        bestFeatLabel = labels[bestFeat]
        returnTree = {bestFeatLabel:{}}
        del(labels[bestFeat])
        featValues = [data[bestFeat] for data in dataSet]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            subLabels = labels[:]
            returnTree[bestFeatLabel][value] = self.createTree(self.splitDataSet(dataSet, bestFeat, value), subLabels)
        return returnTree

    def plotNode(self, nodeTxt, centerPt, parentPt, nodeType):
        self.createPlot().ax1.annotate(nodeTxt,
                                     xy=parentPt, xycoords='axes fraction',
                                     xytext=centerPt, textcoords='axes fraction',
                                     va='center', ha='center', bbox=nodeType, arrowprops=self.arrow_args)

    def createPlot(self):
        fig = plt.figure(1, facecolor='white')
        fig.clf()
        ax1 = plt.subplot(111, frameon=False)
        self.plotNode(u'decision node', (0.5, 0.1), (0.1, 0.5), self.decisionNode)
        self.plotNode(u'leaf node', (0.8, 0.1), (0.3, 0.8), self.leafNode)
        plt.show()

if __name__ == '__main__':
    dtree = dTree()
    dataSet, labels = dtree.createDataSet()
    myTree = dtree.createTree(dataSet, labels)
    # print myTree
    dtree.createPlot()