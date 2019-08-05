#include <math.h>

#include "util.h"

#ifndef __CONNECT_H__
#define __CONNECT_H__


using std::vector;

class BaseConnect {
	int in_size, out_size;
protected:
	int alp;
public:
	BaseConnect(int in, int out, int alp): 
		in_size(in), out_size(out), alp(alp) 
		{}
	int inSize() { return in_size; }
	int outSize() { return out_size; }
	virtual void forward (Layer& L1, Layer& L2) = 0;
	virtual void backward(Layer& L2, Layer& L1) = 0;
};

// y = f(W, X), dy/dwi = df(W, X, i)
template<class Fn>
class FullConnect : public BaseConnect {
	vector<vector<double> > W;
public:
	FullConnect(int in, int out, int alp): 
		BaseConnect(in, out, alp), W(out, vector<double>(in)) 
		{}

	virtual void forward (Layer& L1, Layer& L2) {
		for(int i = 0; i < L2.size(); i++)
			L2[i] = Fn.f(W[i], L1);
	}
	virtual void backward(Layer& L2, Layer& L1) {
		for(int i = 0; i < L2.size(); i++)
		for(int j = 0; j < L1.size(); j++)
			W[i][j] -= alpha * L2[i] * Fn.dfdw(W[i], L1, j);
		for(int i = 0; i < L2.size(); i++)
		for(int j = 0; j < L1.size(); j++)
			L1[j] += L2[i] * Fn.dfdx(W[i], L1, j);
	}
};

template<class f, class df>
class NotFullConnect : public BaseConnect {
	struct Node {
		int ind;
		double val;
	};
	vector<vector<Node> > W;

public:
	NotFullConnect(int in, int out, int alp): 
		BaseConnect(in, out, alp), W(out)
		{}

	virtual void forward (Layer& L1, Layer& L2) {
		for(int i = 0; i < L2.size(); i++)
			L2[i] = Fn.f(W[i], L1);

	}
	virtual void backward(Layer& L2, Layer& L1) {
		for(int i = 0; i < L2.size(); i++)
		for(auto& nd: W[i])
			nd.val -= alpha * L2[i] * Fn.dfdw(W[i], L1, nd.ind);
		for(int i = 0; i < L2.size(); i++)
		for(auto& nd: W[i])
			L1[nd.ind] += L2[i] * Fn.dfdx(W[i], L1, nd.ind);

	}
};

template<class Fn>
class Activation : public BaseConnect {
public:
	Activation(int sz): 
		BaseConnect(sz, sz, 0) 
		{}

	virtual void forward (Layer& L1, Layer& L2) {
		for(int i = 0; i < L1.size(); i++)
			L2[i] = Fn.f(L1[i]);
	}
	virtual void backward(Layer& L2, Layer& L1) {
		for(int i = 0; i < L1.size(); i++)
			L1[i] = Fn.dfdx(L2[i]);
	}
};

#endif



