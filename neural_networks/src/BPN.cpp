#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include "BPN.h"

using namespace std;

BPN::BPN(int _numIn, int _numOut, int _numHidden, double rate = 0.05) {
    this->numIn = _numIn;
    this->numOut = _numOut;
    this->numHidden = _numHidden;
    this->dataIn = vector<double>(_numIn, 0);
    this->dataHidden = vector<double>(_numHidden, 0);
    this->goal = vector<double>(_numOut, 0);
    this->output = vector<double>(_numOut, 0);
    for (int i = 0; i < _numHidden; i++) {
        this->w0.push_back(initRandomVector(_numIn));
    }
    for (int i = 0; i < _numOut; i++) {
        this->w1.push_back(initRandomVector(_numHidden));
    }
    this->biasInHid = initRandomVector(_numHidden);
    this->biasHidOut = initRandomVector(_numOut);
    this->errorHidden = vector<double>(_numHidden, 0);
    this->errorOut = vector<double>(_numOut, 0);
    this->errorHidSum = 0.0;
    this->errorOutSum = 0.0;
    this->learningRate = rate;
    this->count = 0;
}

double BPN::sigmoid(double x) {
    return 1.0 / (1 + exp(-x));
}

vector<double> BPN::initRandomVector(int length) {
    vector<double> temp;
    random_device rd;
    uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < length; i++) {
        temp.push_back(dist(rd));
    }
    return temp;
}

vector<double> BPN::calculate(vector<double> vectorIn,
    vector<double> vectorOut) {
    this->dataIn = vectorIn;
    this->goal = vectorOut;
    for (int i = 0; i < this->numHidden; i++) {
        this->dataHidden[i] = this->biasInHid[i];
        for (int j = 0; j < this->numIn; j++) {
            this->dataHidden[i] += this->w0[i][j] * this->dataIn[j];
        }
        this->dataHidden[i] = sigmoid(this->dataHidden[i]);
    }
    for (int i = 0; i < this->numOut; i++) {
        this->output[i] = this->biasHidOut[i];
        for (int j = 0; j < this->numHidden; j++) {
            this->output[i] += this->w1[i][j] * this->dataHidden[j];
        }
        this->output[i] = sigmoid(this->output[i]);
    }
    return this->output;
}

double BPN::errorDetect() {
    this->errorOutSum = 0.0;
    for (int i = 0; i < this->numOut; i++) {
        this->errorOut[i] = this->output[i] * (1 - this->output[i]) * (this->goal[i] - this->output[i]);
        this->errorOutSum += fabs(this->errorOut[i]);
    }
    this->errorHidSum = 0.0;
    for (int i = 0; i < this->numHidden; i++) {
        this->errorHidden[i] = 0.0;
        for (int j = 0; j < this->numOut; j++) {
            this->errorHidden[i] += this->w1[j][i] * this->errorOut[j];
        }
        this->errorHidden[i] *= this->dataHidden[i] * (1 - this->dataHidden[i]);
        this->errorHidSum += fabs(this->errorHidden[i]);
    }
    return this->errorOutSum;
}

void BPN::updateWeight() {
    for (int i = 0; i < this->numOut; i++) {
        for (int j = 0; j < this->numHidden; j++) {
            this->w1[i][j] += this->learningRate * this->errorOut[i] * this->dataHidden[j];
        }
    }
    for (int i = 0; i < this->numHidden; i++) {
        for (int j = 0; j < this->numIn; j++) {
            this->w0[i][j] += this->learningRate * this->errorHidden[i] * this->dataIn[j];
        }
    }
}

void BPN::updateBias() {
    for (int i = 0; i < this->numOut; i++) {
        this->biasHidOut[i] += this->learningRate * this->errorOut[i];
    }
    for (int i = 0; i < this->numHidden; i++) {
        this->biasInHid[i] += this->learningRate * this->errorHidden[i];
    }
}

bool BPN::finish() {
    return this->count <= 0;
}

void BPN::learn(vector<vector<double> > trainIn,
    vector<vector<double> > trainOut, int maxIterator) {
    this->count = maxIterator;
    while (!this->finish()) {
        cout << this->count << " iterations left." << endl;
        for (int i = 0; i < trainIn.size(); i++) {
            this->calculate(trainIn[i], trainOut[i]);
            this->errorDetect();
            this->updateWeight();
            this->updateBias();
        }
        this->count -= 1;
    }
    cout << "Finish " << maxIterator << " learning(s)." << endl;
}