#include <iostream>

int main()
{
    using namespace std;
    struct ok {
        int a;
        int b;
    };
    cout << ok().a << endl;
    cout << ok().b << endl;
}