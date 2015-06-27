import random, math, numpy

def readData(filePath):
    f = open(filePath)
    data = []
    for line in f:
        string = line[:line.find('\r')]
        data.append(map(int, string.split(',')))
    return data

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

class BPN:
    """BP network class."""

    def __init__(self, _numIn, _numOut, _numHidden, rate = 0.05):
        self.numIn = _numIn
        self.numOut = _numOut
        self.numHidden = _numHidden

        self.dataIn = numpy.zeros(numIn)
        self.dataHidden = numpy.zeros(numHidden)
        self.goal = numpy.zeros(numOut)
        self.output = numpy.zeros(numOut)

        self.w0 = numpy.random.uniform(-1, 1, (numHidden, numIn))
        self.w1 = numpy.random.uniform(-1, 1, (numOut, numHidden))
        self.biasInHid = numpy.random.uniform(-1, 1, numHidden)
        self.biasHidOut = numpy.random.uniform(-1, 1, numOut)

        self.errorHidden = numpy.zeros(numHidden)
        self.errorOut = numpy.zeros(numOut)
        self.errorHidSum = 0.0
        self.errorOutSum = 0.0

        self.learningRate = rate
        self.count = 0
        pass

    def calculate(self, vectorIn, vectorOut):
        self.dataIn = numpy.array(vectorIn)
        self.goal = numpy.array(vectorOut)

        self.dataHidden = self.biasInHid.copy()
        self.dataHidden += numpy.dot(self.dataIn, self.w0.T)
        self.dataHidden = 1 / (1 + numpy.exp(-self.dataHidden))

        self.output = self.biasHidOut.copy()
        self.output += numpy.dot(self.dataHidden, self.w1.T)
        self.output = 1 / (1 + numpy.exp(-self.output))
        return self.output

    def errorDetect(self):
        self.errorOut = self.output * (1 - self.output) * (self.goal - self.output)
        self.errorOutSum = numpy.sum(numpy.abs(self.errorOut))

        self.errorHidden = numpy.dot(self.errorOut, self.w1)
        self.errorHidden *= self.dataHidden * (1 - self.dataHidden)
        self.errorHidSum = numpy.sum(numpy.abs(self.errorHidden))
        pass

    def updateWeight(self):
        template = self.errorOut.reshape(self.numOut, 1) * self.dataHidden.reshape(1, self.numHidden)
        self.w1 += self.learningRate * template
        template = self.errorHidden.reshape(self.numHidden, 1) * self.dataIn.reshape(1, self.numIn)
        self.w0 += self.learningRate * template

    def updateBias(self):
        self.biasHidOut += self.learningRate * self.errorOut
        self.biasInHid += self.learningRate * self.errorHidden

    def finish(self):
        """Determine whether to stop learning or not."""
        return self.count <= 0
        
    def learn(self, trainIn, trainOut, maxIteration):
        print "Number of input: " + str(numIn)
        print "Number of output: " + str(numOut)
        print "Number of hidden: " + str(numHidden)
        self.count = maxIteration
        while not self.finish():
            print str(self.count) + " iterations left."
            for i in xrange(len(trainIn)):
                self.calculate(trainIn[i], trainOut[i])
                self.errorDetect()
                self.updateWeight()
                self.updateBias()
            self.count -= 1
        print "Finish " + str(maxIteration) + " learning(s)."
        pass

if __name__ == "__main__":

    trainFile = 'digitstra.txt'
    trainSet = readData(trainFile)
    trainNum = len(trainSet)
    trainInput = []
    trainOutput = []
    for item in trainSet:
        attribute = item.pop()
        vectorOut = [0 for i in xrange(10)]
        vectorOut[attribute] = 1
        trainOutput.append(vectorOut)
        trainInput.append(item)

    numIn = len(trainInput[0])
    numOut =len(trainOutput[0])
    numHidden = int(numpy.sqrt(numIn * numOut)) + 1
    learningRate = 0.05
    
    bpn = BPN(numIn, numOut, numHidden, learningRate)
    bpn.learn(trainInput, trainOutput, 20)

    testFile = 'digitstest.txt'
    testSet = readData(testFile)
# The last record is incorrect.
    testSet.pop()
    testNum = len(testSet)
    testInput = []
    testOutput = []
    for item in testSet:
        attribute = item.pop()
        vectorOut = [0 for i in xrange(10)]
        vectorOut[attribute] = 1
        testOutput.append(vectorOut)
        testInput.append(item)

    correctNum = testNum
    for i in xrange(testNum):
        testResult = list(bpn.calculate(testInput[i], testOutput[i]))
        if map(round, testResult) != testOutput[i]:
            correctNum -= 1
#        print "Test case: " + str(i + 1)
#        print "Result: " + str(map(round, testResult)) + "    Expect: " + str(testOutput[i]) + '\n'
    print "Match rate: " + str(float(correctNum) / testNum * 100) + '%   [' + str(correctNum) + '/' + str(testNum) + ']'
