/* 
    author: Jasper Xu
    trainData.h:
        Declaration of class TrainData,
        a class related to geting data from training file.
*/

#ifndef _TRAIN_DATA_H_
#define _TRAIN_DATA_H_

class TrainData
{
  public:
    TrainData(const std::string filename);
    void getTopology(std::vector<unsigned> &topology);
    bool ifEof();
    unsigned getInputVals(std::vector<double> &inputVals);
    unsigned getTargetVals(std::vector<double> &targetVals);

  private:
    std::ifstream m_dataFile;
};

#endif // _TRAIN_DATA_H_