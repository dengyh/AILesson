import math

FILE = 'dataset.txt'

ATTRIBUTES = {
    'handicapped-infants' : ['y','n','?'],
    'water-project-cost-sharing' : ['y','n','?'],
    'adoption-of-the-budget-resolution' : ['y','n','?'],
    'physician-fee-freeze' : ['y','n','?'],
    'el-salvador-aid' : ['y','n','?'],
    'religious-groups-in-schools' : ['y','n','?'],
    'anti-satellite-test-ban' : ['y','n','?'],
    'aid-to-nicaraguan-contras' : ['y','n','?'],
    'mx-missile' : ['y','n','?'],
    'immigration' : ['y','n','?'],
    'synfuels-corporation-cutback' : ['y','n','?'],
    'education-spending' : ['y','n','?'],
    'superfund-right-to-sue' : ['y','n','?'],
    'crime' : ['y','n','?'],
    'duty-free-exports' : ['y','n','?'],
    'export-administration-act-south-africa' : ['y','n','?'],
    }

RESULT = {
    'positive' : 'democrat',
    'negative' : 'republican',
}

class Example:
    def __init__(self, input):
        attrs = input.split(',')
        attrValues = attrs[1:]
        count = 0
        self.attributes = {}
        for item in ATTRIBUTES:
            self.attributes[item] = attrValues[count]
            count += 1
        self.result = attrs[0]

class TreeNode:
    def __init__(self, attribute):
        self.attribute = attribute
        self.offspring = {}

class Classification:
    def __init__(self, value):
        self.value = value

def pluralityValue(parentExamples):
    result = {}
    maxClass = ''
    maxValue = 0
    for item in parentExamples:
        if item.result not in result:
            result[item.result] = 1
        else:
            result[item.result] = result[item.result] + 1
        if result[item.result] > maxValue:
            maxClass = item.result
            maxValue = result[item.result]
    return Classification(maxClass)

def checkHasSameClassification(examples):
    last = None
    for item in examples:
        if last:
            if item.result != last.result:
                return False
        last = item
    return True

def log2(value):
    return math.log(value) / math.log(2)

def equal(valueA, valueB):
    if abs(valueA - valueB) < 0.00001:
        return True

def getEntropy(value):
    if equal(value, 1.0) or equal(value, 0.0):
        return 0.0
    return -(value * log2(value) + (1 - value) * log2(1 - value))

def getImportantValue(attribute, examples):
    remainder = 0
    totalLength = len(examples)
    positiveLength = 0
    total = {}
    positive = {}
    for attrValue in ATTRIBUTES[attribute]:
        total[attrValue] = 0
        positive[attrValue] = 0
    for item in examples:
        total[item.attributes[attribute]] += 1
        if item.result == RESULT['positive']:
            positive[item.attributes[attribute]] += 1
            positiveLength += 1
    # print total
    # print positive
    # print attribute
    for attrValue in ATTRIBUTES[attribute]:
        if equal(total[attrValue], 0.0):
            continue
        remainder += total[attrValue] * 1.0 / totalLength * getEntropy(positive[attrValue] * 1.0 / total[attrValue])
    return getEntropy(positiveLength * 1.0 / totalLength) - remainder;

def decisionTreeLearning(examples, attributes, parentExamples):
    if len(examples) == 0:
        return pluralityValue(parentExamples)
    if checkHasSameClassification(examples):
        return Classification(examples[0].result)
    if len(attributes) == 0:
        return pluralityValue(parentExamples)
    attribute = ''
    attributeImport = 0
    for attr in attributes:
        importValue = getImportantValue(attr, examples)
        if importValue > attributeImport:
            attributeImport = importValue
            attribute = attr
    tree = TreeNode(attribute)
    attributes.remove(attribute)
    for value in ATTRIBUTES[attribute]:
        tempExamples = []
        for item in examples:
            if item.attributes[attribute] == value:
                tempExamples.append(item)
        subtree = decisionTreeLearning(tempExamples, attributes, examples)
        tree.offspring[value] = subtree
    attributes.append(attribute)
    return tree

def findPath(example, tree):
    if isinstance(tree, Classification):
        return tree.value
    return findPath(example, tree.offspring[example.attributes[tree.attribute]])


def inputAndInitilizeTree():
    attributeList = []
    exampleList = []
    testList = []
    for item in ATTRIBUTES:
        attributeList.append(item)
    testCase = int(raw_input("Please select the number of the learning data set: "))
    fin = open(FILE, 'r')
    index = 1
    for line in fin:
        if index > testCase:
            testList.append(Example(line.split('\r')[0].split('\n')[0]))
        else:
            exampleList.append(Example(line.split('\r')[0].split('\n')[0]))
        index += 1
    fin.close()
    return decisionTreeLearning(exampleList, attributeList, exampleList), testList, exampleList

def testAndOutput(learningData, testData, decisionTree):
    correct = 0.0
    for item in testData:
        if item.result == findPath(item, decisionTree):
            correct += 1
    print '%d datasets for learning, %d datasets for test, the correct rate is %.4f' % (len(learningData),
        len(testData), correct / len(testData))

def main():
    decisionTree, testData, learningData = inputAndInitilizeTree()
    testAndOutput(learningData, testData, decisionTree)

main()