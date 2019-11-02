#include <assert.h>
#include <vector>
#include <algorithm>
#include <stdio.h>

#include "util.h"
#include "Connect.h"
#include "FullConnect.h"
#include "Activation.h"

#ifndef __MODEL_H__
#define __MODEL_H__

template<class Label>
void MyErrCal(const Label& label, Layer& L, Layer& D) {
    for(int i = 0; i < L.size(); i++) {
        if(i == label) {
            D[i] = 3 * (L[i] - 1);
        }
        else
            D[i] = L[i];
    }
}


template <class Label>
class Model {
	std::vector<Layer> layers, dlayers;
	std::vector<BaseConnect*> connects;
    function<void(const Label&, Layer&, Layer&)> ErrCal;
public:
    Model(function<void(const Label&, Layer&, Layer&)> errCal = MyErrCal<Label>): 
        ErrCal(errCal) 
        {}
	void AddLayer(BaseConnect* c) {
		assert(layers.empty() or layers.back().size() == c->inSize());
        if(layers.empty()) {
            layers.push_back(Layer(c->inSize()));
		    dlayers.push_back(Layer(c->outSize()));
        }

		layers.push_back(Layer(c->outSize()));
		dlayers.push_back(Layer(c->outSize()));
		connects.push_back(c);
	}
    template<class Data>
	void Forward(const Data& data) {
        for(int i = 0; i < data.size(); i++)
            layers[0][i] = data[i];
		for(int i = 0; i < connects.size(); i++) {
			connects[i]->Forward(layers[i], layers[i + 1]);
		}
	}
	void Backward() {
		for(int i = connects.size() - 1; i >= 0; i--) {
			connects[i]->Backward(dlayers[i + 1], dlayers[i], layers[i]);
		}
	}
	void Train(const Data& data, const Label& label) {
        SHOW("Forward");
		Forward (data);
        SHOW("ErrCal");
		ErrCal	(label, layers.back(), dlayers.back());
        SHOW("Backward");
		Backward();
	}
	const Layer& getResult() const {
		return layers.back();
	}
    const Layer& PredictResult(const Data& data) {
		Forward(data);
		const Layer& res = getResult();
        return res;
    }
	Label _Predict(const Layer& res) const {
		auto it = std::max_element(res.begin(), res.end());
		return it - res.begin();
	}
	Label Predict(const Data& data) const {
		const Layer& res = PredictResult(data);
		return _Predict(res);
	}
};


template<class Label>
double Loss(Model<Label>& model, const Data& data, const Label& label) {
    auto& res = model.PredictResult(data);
    double ret = 0;
    for(int i = 0; i < res.size(); i++) {
        double er = (i == label ? 3 * (res[i] - 1) : res[i]);
        ret += er * er;
    }
    return ret;
}


template<class Label>
double Loss(const Layer& res, const Data& data, const Label& label) {
    double ret = 0;
    for(int i = 0; i < res.size(); i++) {
        double er = (i == label ? res[i] - 1 : res[i]);
        ret += er * er;
    }
    return ret;
}


#endif
