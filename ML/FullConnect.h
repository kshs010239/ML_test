#include "Connect.h"
#include "util.h"
#include <numeric>
#include <math.h>
//#include <algorithm.h>

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
    static void set_dfdx_to_L1(const vector<double>& W, Layer& X, const Layer& Xt, double pre_dfdx) {
        for(int i = 0; i < X.size(); i++)
            X[i] += pre_dfdx * W[i]; 
    }
};

struct Multipy {
    static double L(double x) {
        return 1.0 / (1 + exp(-x));
    }
    static double f(const vector<double>& W, const Layer& X) {
        double ret = 1;
        double balance = 2.0 / W.size();
        
        vector<double> LW = W;
        for(auto &w: LW) {
            assert(w < 1e9 and w > -1e9);
            w = L(w) * balance;
        }

        for(int i = 0; i < W.size(); i++)
            ret *= pow(X[i], LW[i]);
        assert(ret < 1 - 1e-09 and ret > 1e-09);
        return ret;
    }
    static void set_dfdw_to_W(vector<double>& W, const Layer& X, double alp, double pre_dfdx) {
        double delta = alp * pre_dfdx;
        double balance = 2.0 / W.size();

        vector<double> LW = W;
        for(auto &w: LW)
            w = L(w) * balance;

        for(int i = 0; i < W.size(); i++)
            delta *= pow(X[i], LW[i]);
        for(int i = 0; i < W.size(); i++) {
            W[i] -= delta * balance * (1 - LW[i]) * LW[i] * log(X[i]);
        }
    }
    static void set_dfdx_to_L1(const vector<double>& W, Layer& X, const Layer& Xt, double pre_dfdx) {
        double delta = pre_dfdx;
        double balance = 2.0 / W.size();

        vector<double> LW = W;
        for(auto &w: LW)
            w = L(w) * balance;

        for(int i = 0; i < W.size(); i++)
            delta *= pow(Xt[i], LW[i]);
        for(int i = 0; i < W.size(); i++)
            X[i] += delta * LW[i] / Xt[i];
    }
};
