#include "Connect.h"
#include "util.h"
#include <numeric>

struct Linear {
    static double f(const vector<double>& W, const Layer& X) {
        double ret = 0;
        for(int i = 0; i < W.size(); i++)
            ret += W[i] * X[i];
        return ret;
    }
    static double dfdw(const vector<double>& W, const Layer& X, int ind) {
        return X[ind];
    }
    static double dfdx(const vector<double>& W, const Layer& X, int ind) {
        return W[ind];
    }
};
