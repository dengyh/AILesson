#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>

#include "BPN.h"

using namespace std;

const string TRAIN_DATA = "digitstra.txt";
const string TEST_DATA = "digitstest.txt";
const double LEARN_RATE = 0.05;

int convertStringToInteger(string s) {
    int num = 0;
    for (int i = 0; i < s.length(); i++) {
        num = num * 10 + (s[i] - '0');
    }
    return num;
}

vector<vector<double> > readData(const char* filePath) {
    fstream fin(filePath);
    vector<vector<double> > datas;
    string line;
    while (getline(fin, line)) {
        string num;
        vector<double> data;
        for (int i = 0; i < line.length(); i++) {
            if (line[i] >= '0' && line[i] <= '9') {
                num += line[i];
            } else {
                data.push_back(convertStringToInteger(num));
                num = "";
            }
        }
        if (num != "") {
            data.push_back(convertStringToInteger(num));
        }
        if (!data.empty()) {
            datas.push_back(data);
        }
    }
    fin.close();
    return datas;
}

void initialize(vector<vector<double> >& trainInput,
    vector<vector<double> >& trainOutput, vector<vector<double> > data) {
    for (int i = 0; i < data.size(); i++) {
        vector<double> vectorOut(10, 0);
        vectorOut[data[i].back()] = 1;
        data[i].erase(data[i].end() - 1);
        trainInput.push_back(data[i]);
        trainOutput.push_back(vectorOut);
    }
}

bool equal(vector<double> c1, vector<double> c2) {
    for (int i = 0; i < c1.size(); i++) {
        if (fabs(c1[i] - c2[i]) > 0.00001) {
            return false;
        }
    }
    return true;
}

int main() {
    vector<vector<double> > trainData, testData, trainInput,
        trainOutput, testInput, testOutput;
    trainData = readData(TRAIN_DATA.c_str());
    testData = readData(TEST_DATA.c_str());
    initialize(trainInput, trainOutput, trainData);
    initialize(testInput, testOutput, testData);
    int numIn = trainInput[0].size();
    int numOut = trainOutput[0].size();
    int numHidden = (int)sqrt(numIn * numOut) + 1;
    BPN bpn(numIn, numOut, numHidden, LEARN_RATE);
    clock_t start = clock();
    bpn.learn(trainInput, trainOutput, 100);
    clock_t end = clock();
    cout << "Total time for learn: " << (end * 1.0 - start) / CLOCKS_PER_SEC << endl;
    int correct = testData.size();
    for (int i = 0; i < testData.size(); i++) {
        vector<double> result = bpn.calculate(testInput[i], testOutput[i]);
        for (int j = 0; j < result.size(); j++) {
            result[j] = (int)(result[j] + 0.5);
        }
        if (!equal(result, testOutput[i])) {
            correct -= 1;
        }
    }
    cout << "Match rate: " << (correct * 1.0 / testData.size()) << "%";
    cout << " [" << correct << " / " << testData.size() << "]" << endl;
    return 0;
}