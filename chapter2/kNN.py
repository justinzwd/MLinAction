from numpy import *
import operator
import matplotlib
from matplotlib import pyplot as plt
from os import listdir

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        sortedClassCount = sorted(classCount.items(),\
                                  key=operator.itemgetter(1),\
                                  reverse = True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet / tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('chapter2/datingTestSet.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],\
                                     datingLabels[numTestVecs:m],3)
        print("the classfier came back with: %d, the real answer is: %d"\
              % (classifierResult,datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is %f" % (errorCount / float(numTestVecs)))

def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(input(\
        "percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix("chapter2/datingTestSet.txt")
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("You will probably like this person:",\
          resultList[classifierResult-1])

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('chapter2/digits/trainingDigits')
    #m是训练文件夹的文件个数
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        #因为文件名都是"0_12.txt"之类的，所以fileStr将是类似于0_12的形式字符串
        fileStr = fileNameStr.split('.')[0]
        #每个文件代表的label，待测数字
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        #第i行填充1024个数字
        trainingMat[i,:] = img2vector('chapter2/digits/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('chapter2/digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('chapter2/digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest,\
                                     trainingMat,hwLabels,3)
        print("the classifier came back with: %d, the real answer is: %d" \
              % (classifierResult,classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))



if __name__ == '__main__':
    # group,labels = createDataSet()
    # print(classify0([0,0],group,labels,3))
    # datingDataMat, datingLabels = file2matrix("chapter2/datingTestSet.txt")
    # print(datingDataMat)
    # print(datingLabels)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
    # plt.show()

    # normMat,ranges,minVals = autoNorm(datingDataMat)
    # print(normMat)
    # print(ranges)
    # print(minVals)

    # datingClassTest()

    # classifyPerson()

    # testVector = img2vector("chapter2/digits/testDigits/0_13.txt")
    # print(testVector[0,0:31])
    # print(testVector[0,32:63])

    handwritingClassTest()
