/*
    author: Jasper Xu
    print.cpp: Implementation of class print
*/

#include <iostream>
#include <vector>
#include <cstdlib>

#include "print.h"

using namespace std;

Print::Print() {}

void Print::printDoubleVectorVals(string label, vector<double> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); i++)
    {
        cout << v[i] << " ";
    }
    cout << endl;
}