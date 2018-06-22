#include <iostream>
#include <cmath>
#include <cstdlib>

using namespace std;

int main()
{
    for (int i = 3000; i >= 1; i--)
    {
        int n1 = int ((2.0 * rand() / double(RAND_MAX)));
        int n2 = int ((2.0 * rand() / double(RAND_MAX)));
        int out = n1 ^ n2; // '^' means XOR

        cout << "in: " << n1 << ".0 " << n2 << ".0" << endl;
        cout << "out: " << out << ".0" << endl;
    }
}
