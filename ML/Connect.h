#include <math.h>

#include "util.h"

#ifndef __CONNECT_H__
#define __CONNECT_H__



class BaseConnect {
	int in_size, out_size;
public:
	double alp;
	BaseConnect(int in, int out, double alp): 
		in_size(in), out_size(out), alp(alp) 
		{}
	int inSize() const { return in_size; }
	int outSize() const { return out_size; }
	virtual void Forward (Layer& L1, Layer& L2) = 0;
	virtual void Backward(Layer& L2, Layer& L1) = 0;
};

// y = f(W, X), dy/dwi = df(W, X, i)
template<class Fn>
class FullConnect : public BaseConnect {
	vector<vector<double> > W;
public:
	FullConnect(int in, int out, std::function< double() >& randomInit, double alp = 0): 
		BaseConnect(in, out, alp), W(out, vector<double>(in)) 
		{   
            for(auto &v: W)
            for(auto &x: v)
                x = randomInit();
        }

	virtual void Forward (Layer& L1, Layer& L2) {
		for(int i = 0; i < L2.size(); i++)
			L2[i] = Fn::f(W[i], L1);
	}
	virtual void Backward(Layer& L2, Layer& L1) {
		for(int i = 0; i < L2.size(); i++)
		for(int j = 0; j < L1.size(); j++)
			W[i][j] -= alp * L2[i] * Fn::dfdw(W[i], L1, j);

        fill(L1.begin(), L1.end(), 0);
		for(int i = 0; i < L2.size(); i++)
		for(int j = 0; j < L1.size(); j++)
			L1[j] += L2[i] * Fn::dfdx(W[i], L1, j);
	}
};

template<class Fn>
class NotFullConnect : public BaseConnect {
	struct Node {
		int ind;
		double val;
	};
	vector<vector<Node> > W;

public:
	NotFullConnect(int in, int out, double alp): 
		BaseConnect(in, out, alp), W(out)
		{}

	virtual void Forward (Layer& L1, Layer& L2) {
		for(int i = 0; i < L2.size(); i++)
			L2[i] = Fn::f(W[i], L1);

	}
	virtual void Backward(Layer& L2, Layer& L1) {
		for(int i = 0; i < L2.size(); i++)
		for(auto& nd: W[i])
			nd.val -= alp * L2[i] * Fn::dfdw(W[i], L1, nd.ind);
        
        fill(L1.begin(), L1.end(), 0);
		for(int i = 0; i < L2.size(); i++)
		for(auto& nd: W[i])
			L1[nd.ind] += L2[i] * Fn::dfdx(W[i], L1, nd.ind);

	}
};

template<class Fn>
class Activation : public BaseConnect {
public:
	Activation(int sz): 
		BaseConnect(sz, sz, 0) 
		{}

	virtual void Forward (Layer& L1, Layer& L2) {
		for(int i = 0; i < L1.size(); i++)
			L2[i] = Fn::f(L1[i]);
	}
	virtual void Backward(Layer& L2, Layer& L1) {
		for(int i = 0; i < L1.size(); i++)
			L1[i] = Fn::dfdx(L2[i]);
	}
};

#endif



