/*
    author: Jasper Xu
    neuron.h: Declaration of Neuron, Layer and Connection
*/

#ifndef _NEURON_H_
#define _NEURON_H_

struct Connection
{
    double weight;
    double deltaWeight;
};

class Neuron;

typedef std::vector<Neuron> Layer;

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
    std::vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_grad;
    static double eta;   // overall net learning rate
    static double alpha; // momentum
};

#endif // _NEURON_H_