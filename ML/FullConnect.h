#include "Connect.h"
#include "util.h"
#include <numeric>
#include <math.h>

struct Linear {
    static double f(const vector<double>& W, const Layer& X) {
        double ret = 0;
        for(int i = 0; i < W.size(); i++)
            ret += W[i] * X[i];
        return ret;
    }
    static void set_dfdw_to_W(vector<double>& W, const Layer& X, double alp, double pre_dfdx) {
        for(int i = 0; i < W.size(); i++) 
            W[i] -= alp * pre_dfdx * X[i];
    }
    static void set_dfdx_to_L2(const vector<double>& W, const Layer& X) {
        for(int i = 0; i < X.size(); i++)
            X[i] += 0; 
    }
};

struct Multipy {
    static double L(double x) {
        return 1.0 / (1 + exp(-x));
    }
    static double f(const vector<double>& W, const Layer& X) {
        double ret = 1;
        for(int i = 0; i < W.size(); i++)
            ret *= pow(X[i], L(W[i]));
        return ret;
    }
    static double pre_dfdw(const vector<double>& W, const Layer& X) {
        double delta = alpha;
        for(int i = 0; i < W.size(); i++)
            delta *= pow(X[i], L(W[i]));
        return delta;
    }
    static double dfdw(const vector<double>& W, const Layer& X, int ind) {
        double ret = 1;
        return delta
    }
    static double dfdx(const vector<double>& W, const Layer& X, int ind) {
        return W[ind];
    }
}
