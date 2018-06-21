// -*- coding: utf-8 -*-
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>

using namespace std;

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
    static double eta; // overall net learning rate
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

  private:
    vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
    double m_recentAverageSmoothingFactor;
};

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
    m_recentAverageError = (m_recentAverageError + m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1);

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
        }

        // set the bias nodes' output to be 1.0
        m_layers.back().back().setOutputVal(1.0);
    }
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

// ***************************************** main *****************************************
int main()
{
    // test1: make a neuron network {3,2,1}

    vector<unsigned> topology;
    topology.push_back(3);
    topology.push_back(2);
    topology.push_back(1);
    Net myNet(topology);

    vector<double> inputVals;
    inputVals.push_back(1.0);
    inputVals.push_back(1.5);
    inputVals.push_back(0.5);
    myNet.feedForward(inputVals);

    // vector<double> targetVals;
    // myNet.backPropagation(targetVals);

    // vector<double> resultVals;
    // myNet.getResults(resultVals);
}