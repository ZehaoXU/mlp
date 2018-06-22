#include <iostream>
#include <cmath>

using namespace std;

int main()
{
    // the target function is y = sin(a)
    // double a & b in range(0,179)
    for (int i = 80000; i >= 1; i--)
    {
        int a1 = int(rand() % 360);
        // int b1 = int(rand() % 360);
        const double pi = 3.14159;
        double a = a1 * pi / 180;
        // double b = b1 * pi / 180;

        double out = sin(a);

        cout << "in: " << a << endl;
        cout << "out: " << out << endl;
    }
}