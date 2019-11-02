#include <math.h>

#include "Connect.h"



struct Sigmoid {
    static double f(double x) {
        return 1.0 / (1 + exp(-x)); 
    }
    static double dfdx(double x) {
        double fx = f(x);
        return fx * (1 - fx);
    }
};

struct ReLU {
    static double f(double x) {
        return x > 0 ? x : 0;
    }
    static double dfdx(double x) {
        return x > 0 ? 1 : 0;
    }
};
