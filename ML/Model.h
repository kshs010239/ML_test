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

class MyErrCal {
public:
	void operator()(uint32_t& label, Layer& L, Layer& D) {
		for(int i = 0; i < L.size(); i++) {
			if(i == label)
				D[i] = 1 - L[i];
			else
				D[i] = L[i];
		}
	}
};

template<class Label, class ErrCal = MyErrCal >
class Model {
	std::vector<Layer> layers, dlayers;
	std::vector<BaseConnect*> connects;
public:
	void AddLayer(BaseConnect* c) {
		assert(layers.empty() or layers.back().size() == c->inSize());

		layers.push_back(Layer(c->outSize()));
		connects.push_back(c);
	}
	void Forward(Data& data) {
        for(int i = 0; i < data.size(); i++)
            layers[0][i] = data[i];

		for(int i = 0; i < connects.size(); i++) {
			connects[i]->Forward(layers[i], layers[i + 1]);
		}
	}
	void Backward(Data& data) {
		for(int i = 0; i < connects.size(); i++) {
			connects[i]->Backward(layers[i + 1], layers[i]);
		}
	}
	void Train(Data& data, Label& label) {
		Forward (data);
		ErrCal	(label, layers.back(), dlayers.back());
		Backward(data);
	}
	const Layer& getResult() {
		return layers.back();
	}
    const Layer& PredictResult(Data& data) {
		Forward(data);
		const Layer& res = getResult();
        return res;
    }
	Label Predict(Data& data) {
		const Layer& res = PredictResult();
		auto it = std::max_element(res.begin(), res.end());
		return it - res.begin();
	}

};



#endif
