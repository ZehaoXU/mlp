/*
    author: Jasper Xu
    neuron.cpp: 
        Definition of member functions in class Neuron
        and  initialization of static variables.
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <cstdlib>

#include "neuron.h"

using namespace std;

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    for (unsigned i = 0; i < numOutputs; i++)
    {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    m_myIndex = myIndex;
}

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


void Neuron::setOutputVal(double val)
{
    m_outputVal = val;
}

double Neuron::getOutputVal() const
{
    return m_outputVal;
}

// return a value in range[0, 1]
double Neuron::randomWeight()
{
    return rand() / double(RAND_MAX);
}

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

double Neuron::transferFunction(double sum)
{
    // use tanh as a transfer function
    // neuron output value in range[-1, 1]
    return tanh(sum);
}

double Neuron::transferFunctionDerivative(double x)
{
    // tanh derivative
    return 1.0 - x * x;
}

void Neuron::calOutputGrad(double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_grad = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

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

void Neuron::updateWeights(Layer &preLayer)
{
    // modify the output weight in the previous Layer
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