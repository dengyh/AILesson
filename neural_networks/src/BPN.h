#ifndef BPN_H
#define BPN_H

#include <iostream>
#include <string>
#include <vector>
#include <cmath>

using namespace std;

class BPN {
public:
    BPN(int, int, int, double);
    void learn(vector<vector<double> >,
        vector<vector<double> >, int);
    vector<double> calculate(vector<double>, vector<double>);
private:
    double sigmoid(double);
    vector<double> initRandomVector(int);
    double errorDetect();
    void updateWeight();
    void updateBias();
    bool finish();

    int numIn, numOut, numHidden;
    vector<double> dataIn, dataHidden, goal, output;
    vector<vector<double> > w0, w1;
    vector<double> biasInHid, biasHidOut;
    vector<double> errorHidden, errorOut;
    double errorHidSum, errorOutSum, learningRate;
    int count;
};

#endif