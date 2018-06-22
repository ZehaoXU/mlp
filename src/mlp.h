/*
    author: Jasper Xu
    mlp.h: 
        Declaration of class Net, a topology
        multi-layer perceptron with functions.
*/

#ifndef _MLP_H_
#define _MLP_H_

class Net
{
  public:
    Net(const std::vector<unsigned> &topology);
    void feedForward(const std::vector<double> &inputVals);
    void backPropagation(const std::vector<double> &targetVals);
    void getResults(std::vector<double> &resultVals);
    double getAverageError() const;

  private:
    std::vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
    double m_error;
    double m_averageError;
    static double m_averageSmoothingFactor;
};

#endif // _MLP_H_