/*
    author: Jasper Xu
    mlp.cpp: Implementation of class Net.
*/

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>

#include "neuron.h"
#include "mlp.h"

using namespace std;

// number of samples to calculate average
double Net::m_averageSmoothingFactor = 100.0;

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
        // exclude the bias node
        for (unsigned n = 0; n < m_layers[layerNum].size() - 1; n++)
        {
            m_layers[layerNum][n].feedForward(preLayer);
        }
    }
}

void Net::getResults(vector<double> &resultVals)
{
    resultVals.clear();

    // get all results from output layer
    Layer &outputLayer = m_layers.back();
    // exclude the bias node
    for (unsigned i = 0; i < outputLayer.size() - 1; i++)
    {
        resultVals.push_back(outputLayer[i].getOutputVal());
    }
}

void Net::backPropagation(const vector<double> &targetVals)
{
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    // m_error is overall net error: RMS root mean square error
    // exclude the bias layer
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
    // exclude the bias layer
    for (unsigned i = 0; i < outputLayer.size() - 1; i++)
    {
        outputLayer[i].calOutputGrad(targetVals[i]);
    }

    // calculate gradients on hidden layers, from back to front
    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; layerNum--)
    {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        // include bias node
        for (unsigned i = 0; i < hiddenLayer.size(); i++) 
        {
            hiddenLayer[i].calHiddenGrad(nextLayer);
        }
    }

    // update the connection weight from the output layer to the first hidden layer
    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; layerNum--)
    {
        Layer &currentLayer = m_layers[layerNum];
        Layer &preLayer = m_layers[layerNum - 1];

        // inlcude the bias layer
        for (unsigned i = 0; i < currentLayer.size(); i++)
        {
            currentLayer[i].updateWeights(preLayer);
        }
    }
}

double Net::getAverageError() const
{
    return m_averageError;
}