#include <assert.h>
#include <vector>
#include <algorithm>

#include "Connect.h"
#include "util.h"

#ifndef __MODEL_H__
#define __MODEL_H__

class ErrCal {
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

template<class Label, class errCal = ErrCal >
class Model {
	std::vector<Layer> layers, dlayers;
	std::vector<BaseConnect> connects;
public:
	void addLayer(BaseConnect& c) {
		assert(layers.back().size() == c.inSize());

		layers.push_back(Layer(c.outSize()));
		connects.push_back(c);
	}
	void forward(Data& data) {
		for(int i = 0; i < connects.size(); i++) {
			connects[i].forward(layers[i], layers[i + 1]);
		}
	}
	void backward(Data& data) {
		for(int i = 0; i < connects.size(); i++) {
			connects[i].backward(dlayers[i + 1], dlayers[i]);
		}
	}
	void train(Data& data, Label& label) {
		forward (data);
		errCal	(label, layers.back(), dlayers.back());
		backward(data);
	}
	const Layer& getResult() {
		return layers.back();
	}
	Label predict(Data& data) {
		forward(data);
		const Layer& res = getResult();
		auto it = std::max_element(res.begin(), res.end());
		return it - res.begin();
	}

};



#endif
