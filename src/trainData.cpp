/*
    author: Jasper Xu
    trainData.cpp: Implementation of class TrainData.
*/

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <sstream>

#include "trainData.h"

using namespace std;

TrainData::TrainData(const string filename)
{
    m_dataFile.open(filename.c_str());
}

void TrainData::getTopology(vector<unsigned> &topology)
{
    string line;
    cout << "[INFO] Please input the topology of MLP "
            "(from input layer to outputer layer and separated by space):"
         << endl;
    getline(cin, line);
    stringstream ss(line);
    while (!ss.eof())
    {
        unsigned i;
        ss >> i;
        topology.push_back(i);
    }
}

bool TrainData::ifEof()
{
    return m_dataFile.eof();
}

// return size of input values
unsigned TrainData::getInputVals(vector<double> &inputVals)
{
    inputVals.clear();

    string line;
    string label;

    // input line look like: 'in: 1.0 0.0 0.5...'
    getline(m_dataFile, line);
    stringstream ss(line);
    ss >> label;
    if (label.compare("in:") == 0)
    {
        double val;
        while (ss >> val)
        {
            inputVals.push_back(val);
        }
    }

    return inputVals.size();
}

// return size of target values
unsigned TrainData::getTargetVals(vector<double> &targetVals)
{
    targetVals.clear();

    string line;
    string label;

    // output line look like: 'out: 1.0 2.0 ...'
    getline(m_dataFile, line);
    stringstream ss(line);
    ss >> label;
    if (label.compare("out:") == 0)
    {
        double val;
        while (ss >> val)
        {
            targetVals.push_back(val);
        }
    }

    return targetVals.size();
}