#include <vector>
#include <iostream>
#include <stdint.h>
#include <functional>

#ifndef __UTIL_H__
#define __UTIL_H__

using std::vector;
using std::cout;
using std::endl;
using std::ostream;
using std::function;
typedef vector<double> Layer;
typedef Layer Data;

#define SHOW(x) do { cout << (#x) << ": " << x << endl; } while(0);

template<class T>
std::ostream& operator<<(std::ostream& out, const vector<T>& v) {
    for(auto &it: v)
        out << it << ' ';
    out << std::endl;
    return out;
}


namespace Random {

double Random(double Max = 1, double Min = 0) {
    double r = (double)rand();
    double d = Max - Min;
    return r / RAND_MAX * d + Min;
}

int Randint(int Max, int Min = 0) {
    int r = rand();
    int d = Max - Min;
    int ret = (r % d) + Min;
    return ret;
}

} // namespace random
#endif
