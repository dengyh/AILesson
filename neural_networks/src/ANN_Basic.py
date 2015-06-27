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
		
        self.dataIn = range(numIn)
        self.dataHidden = range(numHidden)
        self.goal = range(numOut)
        self.output = range(numOut)
		
        self.w0 = [[random.uniform(-1, 1) for i in xrange(numIn)] for j in xrange(numHidden)]
        self.w1 = [[random.uniform(-1, 1) for i in xrange(numHidden)] for j in xrange(numOut)]
        self.biasInHid = [random.uniform(-1, 1) for i in xrange(numHidden)]
        self.biasHidOut = [random.uniform(-1, 1) for i in xrange(numOut)]
		
        self.errorHidden = range(numHidden)
        self.errorOut = range(numOut)
        self.errorHidSum = 0.0
        self.errorOutSum = 0.0
		
        self.learningRate = rate
        self.count = 0
        pass

    def calculate(self, vectorIn, vectorOut):
        self.dataIn = vectorIn
        self.goal = vectorOut
        for i in xrange(self.numHidden):
            self.dataHidden[i] = self.biasInHid[i]
            for j in xrange(self.numIn):
                self.dataHidden[i] += self.w0[i][j] * self.dataIn[j]
            self.dataHidden[i] = sigmoid(self.dataHidden[i])
        for i in xrange(self.numOut):
            self.output[i] = self.biasHidOut[i]
            for j in xrange(self.numHidden):
                self.output[i] += self.dataHidden[j] * self.w1[i][j]
            self.output[i] = sigmoid(self.output[i])
        return self.output

    def errorDetect(self):
        self.errorOutSum = 0.0
        for i in xrange(self.numOut):
            self.errorOut[i] = self.output[i] * (1 - self.output[i]) * (self.goal[i] - self.output[i])
            self.errorOutSum += abs(self.errorOut[i])
        self.errorHidSum = 0.0
        for i in xrange(self.numHidden):
            self.errorHidden[i] = 0
            for j in xrange(self.numOut):
                self.errorHidden[i] += self.w1[j][i] * self.errorOut[j]
            self.errorHidden[i] *= self.dataHidden[i] * (1 - self.dataHidden[i])
            self.errorHidSum += abs(self.errorHidden[i])
        return self.errorOutSum

    def updateWeight(self):
        for i in xrange(self.numOut):
            for j in xrange(self.numHidden):
                self.w1[i][j] += self.learningRate * self.errorOut[i] * self.dataHidden[j]
        for i in xrange(self.numHidden):
            for j in xrange(self.numIn):
                self.w0[i][j] += self.learningRate * self.errorHidden[i] * self.dataIn[j]

    def updateBias(self):
        for i in xrange(self.numOut):
            self.biasHidOut[i] += self.learningRate * self.errorOut[i]
        for i in xrange(self.numHidden):
            self.biasInHid[i] += self.learningRate * self.errorHidden[i]

    def finish(self):
        """Determine whether to stop learning or not."""
        return self.count <= 0
        
    def learn(self, trainIn, trainOut, maxIterator):
        print "Number of input: " + str(numIn)
        print "Number of output: " + str(numOut)
        print "Number of hidden: " + str(numHidden)
        self.count = maxIterator
        while not self.finish():
            print str(self.count) + " iterations left."
            for i in xrange(len(trainIn)):
                self.calculate(trainIn[i], trainOut[i])
                self.errorDetect()
                self.updateWeight()
                self.updateBias()
            self.count -= 1
        print "Finish " + str(maxIterator) + " learning(s)."
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
    bpn.learn(trainInput, trainOutput, 10)

    testFile = 'digitstest.txt'
    testSet = readData(testFile)
# The last record is incorrect.
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
        testResult = bpn.calculate(testInput[i], testOutput[i])
        if map(round, testResult) != testOutput[i]:
            correctNum -= 1
#        print "Test case: " + str(i + 1)
#        print "Result: " + str(map(round, testResult)) + "    Expect: " + str(testOutput[i]) + '\n'
    print "Match rate: " + str(float(correctNum) / testNum * 100) + '%   [' + str(correctNum) + '/' + str(testNum) + ']'
