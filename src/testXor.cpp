/*
    author: Jasper Xu
    textXor.cpp: text the mlp with xor problem.
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <cstdlib>

#include "neuron.h"
#include "mlp.h"
#include "print.h"
#include "trainData.h"

using namespace std;

int main()
{
    // test1: make a neuron network {3,2,1}
    vector<unsigned> topology;

    // read data file
    TrainData trainData("../trainingSet/xorData.txt");
    if (trainData.ifEof() == 0)
    {
        cout << "[INFO] Successfully read data file." << endl;
    }

    trainData.getTopology(topology);

    Net myNet(topology);

    cout << "[INFO] Traning will begin..." << endl;
    system("pause");

    Print print;

    vector<double> inputVals, targetVals, resultVals;
    int step = 0;

    while (!trainData.ifEof())
    {
        step++;
        cout << endl
             << "Step " << step << endl;

        // getInputVals returns the number of input values
        if (trainData.getInputVals(inputVals) != topology[0])
        {
            cout << "[ERROR] Input vector size not equal to input layer size!" << endl;
            cout << inputVals.size() << endl;
            break;
        }
        // show input valus
        print.printDoubleVectorVals("Inputs: ", inputVals);
        myNet.feedForward(inputVals);

        // show feedforward results of myNet
        myNet.getResults(resultVals);
        print.printDoubleVectorVals("Outputs: ", resultVals);

        // train the net with target values
        if (trainData.getTargetVals(targetVals) != topology.back())
        {
            cout << "[ERROR] Target vector size not equal to output layer size!" << endl;
            break;
        }
        print.printDoubleVectorVals("Targets: ", targetVals);
        myNet.backPropagation(targetVals);

        cout << "Net average error: " << myNet.getAverageError() << endl;
    }
    system("pause");
    
}