// -*- coding: utf-8 -*-
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;

/********************* print process data ******************/
class Print
{
  public:
    void printDoubleVectorVals(string label, vector<double> &v);
    Print();

  private:
};

// definition of printDoubleVectorVals
void Print::printDoubleVectorVals(string label, vector<double> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); i++)
    {
        cout << v[i] << " ";
    }
    cout << endl;
}

// constructor function
Print::Print() {}

/******************** train data ****************************/
class TrainData
{
  public:
    TrainData(const string filename);
    void getTopology(vector<unsigned> &topology);
    bool ifEof();
    unsigned getInputVals(vector<double> &inputVals);
    unsigned getTargetVals(vector<double> &targetVals);

  private:
    ifstream m_dataFile;
};

// definition of getTargetVals
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

// definition of getInputs
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

// definition of ifEof
bool TrainData::ifEof()
{
    return m_dataFile.eof();
}

// constructor function
TrainData::TrainData(const string filename)
{
    m_dataFile.open(filename.c_str());
}

// definition of getTopology
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

/*************************** start neural net ***************/
struct Connection
{
    double weight;
    double deltaWeight;
};

// declaration of Neuron
class Neuron;
// definition of Layer
typedef vector<Neuron> Layer;

//*********************** definition of Neuron *********************************
class Neuron
{
  public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void feedForward(const Layer &preLayer);
    void setOutputVal(double val);
    double getOutputVal() const;
    void calOutputGrad(double tragetVal);
    void calHiddenGrad(const Layer &nextLayer);
    void updateWeights(Layer &preLayer);

  private:
    static double transferFunction(double sum); // or an inline function is okay
    static double transferFunctionDerivative(double x);
    static double randomWeight();
    double m_outputVal;
    vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_grad;
    static double eta;   // overall net learning rate
    static double alpha; // momentum
};

/* eta:
    overall net learning rate, range [0.0--1.0]
    0.0 - slow learner
    0.2 - medium learner
    1.0 - reckless learner*/
double Neuron::eta = 0.15;

/* alpha:
    momentum, multiplier of last deltaWeight, range [0.0--n]
    0.0 - no momentum
    0.5 - moderate momentum*/
double Neuron::alpha = 0.5;

// definition of updateWeights
void Neuron::updateWeights(Layer &preLayer)
{
    //
    for (unsigned i = 0; i < preLayer.size(); i++) // include the bias neuron
    {
        Neuron &preNeuron = preLayer[i];
        double oldDeltaWeight = preNeuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight =
            // input part 1, modified by the grandient and the train rate 'eta'
            eta * preNeuron.getOutputVal() * m_grad
            // input part 2, momentum of previous delta weight
            + alpha * oldDeltaWeight;

        preNeuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        preNeuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

// definition of calHiddenGrad
void Neuron::calHiddenGrad(const Layer &nextLayer)
{
    double sum = 0.0;

    // sum something just like 'delat' for hiddent layer
    for (unsigned i = 0; i < nextLayer.size() - 1; i++) // exclude the bias neuron
    {
        sum += m_outputWeights[i].weight * nextLayer[i].m_grad;
    }

    // do just as calOutputGrad
    m_grad = sum * Neuron::transferFunctionDerivative(m_outputVal);
}

// definition of calOutputGrad
void Neuron::calOutputGrad(double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_grad = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

// definition of transferFunction
double Neuron::transferFunction(double sum)
{
    // use tanh as a transfer function range[-1,1]
    return tanh(sum);
}

// definition of transferFunctionDerivative
double Neuron::transferFunctionDerivative(double x)
{
    // tanh derivative
    return 1.0 - x * x;
}

// definition of feedForward
// math function inside a neuron
void Neuron::feedForward(const Layer &preLayer)
{
    double sum = 0.0;

    // sum all the neurons' output value in the previous layer
    // include the bias node in the previous layer
    for (unsigned n = 0; n < preLayer.size(); n++)
    {
        // the index need to be figured out
        sum += preLayer[n].m_outputVal * preLayer[n].m_outputWeights[m_myIndex].weight;
    }
    m_outputVal = Neuron::transferFunction(sum);
}

// definition of setOutputVal
void Neuron::setOutputVal(double val)
{
    m_outputVal = val;
}

// definition of getOutputVal
double Neuron::getOutputVal() const
{
    return m_outputVal;
}

// definition of randomWeight
double Neuron::randomWeight()
{
    return rand() / double(RAND_MAX);
}

// definition of constructor function Neuron
Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    for (unsigned i = 0; i < numOutputs; i++)
    {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    m_myIndex = myIndex;
}

// ***************************** definition of Net ********************************
class Net
{
  public:
    Net(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backPropagation(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals);
    double getAverageError() const;

  private:
    vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
    double m_error;
    double m_averageError;
    static double m_averageSmoothingFactor;
};

// number of samples to calculate average
double Net::m_averageSmoothingFactor = 100.0;

// definition of getAverageError
double Net::getAverageError() const
{
    return m_averageError;
}

// definition of getResults
void Net::getResults(vector<double> &resultVals)
{
    resultVals.clear();

    // get all results from output layer
    Layer &outputLayer = m_layers.back();
    for (unsigned i = 0; i < outputLayer.size() - 1; i++) // exclude the bias node
    {
        resultVals.push_back(outputLayer[i].getOutputVal());
    }
}

// definition of backPropagation
void Net::backPropagation(const vector<double> &targetVals)
{
    // calculate overall net error: RMS root mean square error
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    for (unsigned i = 0; i < outputLayer.size() - 1; i++)
    {
        double delta = targetVals[i] - outputLayer[i].getOutputVal();
        m_error += delta * delta;
    }
    m_error = m_error / (outputLayer.size() - 1); // there is a bias in output layer
    m_error = sqrt(m_error);

    // recent average measurement
    m_averageError = (m_averageError + m_averageSmoothingFactor + m_error) / (m_averageSmoothingFactor + 1);

    // calculate output layer gradients
    for (unsigned i = 0; i < outputLayer.size(); i++)
    {
        outputLayer[i].calOutputGrad(targetVals[i]);
    }

    // calculate gradients on hidden layers, from back to front
    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; layerNum--)
    {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (unsigned i = 0; i < hiddenLayer.size(); i++) // include bias node
        {
            hiddenLayer[i].calHiddenGrad(nextLayer);
        }
    }

    // update the connection weight from the output layer to the first hidden layer
    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; layerNum--)
    {
        Layer &currentLayer = m_layers[layerNum];
        Layer &preLayer = m_layers[layerNum - 1];

        for (unsigned i = 0; i < currentLayer.size(); i++)
        {
            currentLayer[i].updateWeights(preLayer);
        }
    }
}

// definition of constructor function Net
Net::Net(const vector<unsigned> &topology)
{
    cout << "[INFO] Start to build a Neuron Net..." << endl;
    unsigned totalLayers = topology.size(); // number of layers
    for (unsigned layerNum = 0; layerNum < totalLayers; layerNum++)
    {
        m_layers.push_back(Layer());

        // output layer dont have numOutputs
        unsigned numOutputs;
        if (layerNum == totalLayers - 1)
            numOutputs = 0;
        else
            numOutputs = topology[layerNum + 1];

        // create a new layer, now try to fill it with certern num of neurons
        // add a bias neuron to each layer
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++) // add bias neuron so '<='
        {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            cout << "Neuron No." << neuronNum << " in layer No."
                 << layerNum << " is built" << endl;
        }

        // set the bias nodes' output to be 1.0
        m_layers.back().back().setOutputVal(1.0);
        cout << "bias node's output value set 1.0" << endl;
    }

    cout << "[INFO] Successfully build a Neuron Net!" << endl;
}

// definition of feedForward
void Net::feedForward(const vector<double> &inputVals)
{
    // check if input values == neurons in input layer
    assert(inputVals.size() == m_layers[0].size() - 1);

    // assign the input values into the input neurons
    for (unsigned i = 0; i < inputVals.size(); i++)
    {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    // forward propagation
    for (unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++)
    {
        Layer &preLayer = m_layers[layerNum - 1];
        for (unsigned n = 0; n < m_layers[layerNum].size() - 1; n++) // exclude the bias node
        {
            m_layers[layerNum][n].feedForward(preLayer);
        }
    }
}

// ************************************* main *****************************************
int main()
{
    // test1: make a neuron network {3,2,1}
    vector<unsigned> topology;

    // read data file
    TrainData trainData("../funcData.txt");
    if (trainData.ifEof() == 0)
    {
        cout << "[INFO] Successfully read data file." << endl;
    }

    trainData.getTopology(topology);

    Net myNet(topology);

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