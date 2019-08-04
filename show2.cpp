#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include <vector>
#include <iomanip>

using namespace std;

#define DB(x) //cout << (#x) << ": " << (x) << endl

template<class T>
ostream& operator<<(ostream& out, vector<T>& v) {
    for(auto &it: v)
        out << it << " ";
    out << endl;
    return out;
}

FILE *fp, *flabel;

int magicNumber, numberOfImages, width, height, magicLabel, numberOfLabels, imgsize;

typedef uint8_t byte;
typedef vector<uint8_t> Image; 
typedef vector<double> Layer;
typedef vector<vector<double> > Mat;
typedef pair<int, int> Pair;




double e = exp(1);

double L(double x) {
    return 1.0 / (1 + exp(-x));
}


double Random(double Min = -0.01, double Max = 0.01) {
    double r = rand();
    double d = Max - Min;
    double ret = r / RAND_MAX * d + Min;
    return ret;
}

int randint(int Max, int Min = 0) {
    int r = rand();
    int d = Max - Min;
    int ret = (r % d) + Min;
    return ret;
}

void setz(Layer& lr) {
    fill(lr.begin(), lr.end(), 0);
}




struct Model {

    double alpha;

    Layer imgdata, hd, hd2, hd3, out, out2;
  
  //---------------------------------------------
    struct LC { // Linear connect
        vector<vector<double> > W;
        double alpha;
        LC(double alpha, int in, int out): alpha(alpha), W(in, vector<double>(out)) {
            for(auto &v: W)
            for(auto &it: v)
                it = Random(); 
        }
        void forward(Layer& L1, Layer& L2) {
            setz(L2);

            for(int j = 0; j < L2.size(); j++)
                for(int i = 0; i < L1.size(); i++)
                    L2[j] += L1[i] * W[i][j];
            
        }
        void backward(Layer& L1, Layer& L2) {

            for(int i = 0; i < L1.size(); i++)
            for(int j = 0; j < L2.size(); j++)
                W[i][j] -= alpha * L2[j] * L1[i]; 
            
            setz(L1);
            for(int i = 0; i < L1.size(); i++)
                for(int j = 0; j < L2.size(); j++)
                    L1[i] += L2[j] * W[i][j];
        }
        void show() {
            for(auto &v: W) {
                for(auto &it: v)
                    cout << it << " ";
                cout << endl;
            }
        }
    };

    struct AndC {
        vector<Pair> from;
        int inSize;
        AndC(int in, int out): inSize(in), from(out - in) {
            for(int i = 0; i < from.size(); i++)
                from[i] = Pair(randint(in), randint(in)); 
        }
        void forward (Layer& L1, Layer& L2) {
            setz(L2);

            for(int i = 0; i < from.size(); i++) {
                Pair& pr = from[i];
                L2[i + inSize] = 2 * L1[pr.first] * L1[pr.second];
            }
            for(int i = 0; i < inSize; i++)
                L2[i] = L1[i];
            DB(L1);
            DB(L2);
        }
        void backward( Layer& L1, Layer& L2) {
            setz(L1);

            for(int i = 0; i < inSize; i++)
                L1[i] += L2[i];
            DB(L1);
            for(int i = 0; i < from.size(); i++) {
                Pair& pr = from[i];
                L1[pr.first] += 2 * L1[pr.second] * L2[i + inSize];
                L1[pr.second] += 2 * L1[pr.first] * L2[i + inSize];
            }
            DB(L1);
        }
    };

    struct LogC {
        void forward(Layer& L1, Layer& L2) {
            for(int i = 0; i < L1.size(); i++)
                L2[i] = L(L1[i]);
        }
        void backward(Layer& L1, Layer& L2) {
            for(int i = 0; i < L1.size(); i++) {
                double x = L(L1[i]);
                L1[i] = x * (1 - x) * L2[i];
            }
        }
    };

    LC lc1, lc2;
    LogC logc;
    AndC andc;

    
    Model(double alpha = 0.01, int sz1 = 100, int szAnd = 100, int sz2 = 10): 
        alpha(alpha), imgdata(imgsize), 
        hd(sz1), hd2(sz1), hd3(szAnd), 
        out(sz2), out2(sz2), 
        lc1(alpha, imgsize, sz1), 
        andc(sz1, szAnd),
        lc2(alpha, szAnd, sz2) 
        {}



    void _predict (Image& img) {
        for(int i = 0; i < imgsize; i++) 
            imgdata[i] = img[i];
        
        lc1.forward(imgdata, hd);
        logc.forward(hd, hd2);
        andc.forward(hd2, hd3); 
        lc2.forward(hd3, out);
        logc.forward(out, out2);
    }
    void _backward(Image& img, uint8_t label) {
        for(int i = 0; i < out2.size(); i++)
            if(i == label) {
                out2[i] = (out2[i] - 1);
            }
        logc.backward(out, out2);
        lc2.backward(hd3, out);
        andc.backward(hd2, hd3);
        logc.backward(hd, hd2);
        lc1.backward(imgdata, hd); 
    }
    void predict (Image& img) {
        _predict(img);
        for(int i = 0; i < 10; i++)
            cout << i << ": " << out2[i] << "\t";
        cout << endl;
    }
    double Er(Image& img, uint8_t label) {
        //cout << "(" << (int)label << ") ";
        _predict(img);
        double ret = 0;
        for(int i = 0; i < 10; i++) {
            double er = (i == label ? 1 - out2[i]: out2[i]);
            ret += er * er;
        }
        return (ret);
    }
    void train (Image& img, uint8_t label) {
        _predict(img);
        _backward(img, label);
    }
    void train (vector<Image>& imgs, vector<uint8_t> labels) {
        cout << "unfinished" << endl;
        exit(0);
    }
    
};


int readint(FILE* p) {
	int ret = 0;
	uint8_t buf[10];
	fread(buf, 1, 4, p);
	for(int i = 0; i < 4; i++)
		ret = ret * 256 + buf[i];
	return ret;
}

int main(int argc, char* argv[]) {
    
	fp = fopen(argv[1], "r");
    flabel = fopen(argv[2], "r");
    srand(time(NULL));


	magicNumber = readint(fp);
	numberOfImages = readint(fp);
	width = readint(fp);
	height = readint(fp);

    magicLabel = readint(flabel);
    numberOfLabels = readint(flabel);


    imgsize = width*height;
    

    vector<vector<uint8_t> > imgs(numberOfImages, vector<uint8_t>(imgsize));
    vector<uint8_t> labels(numberOfLabels);

    for(int imgCnt = 0; imgCnt < numberOfImages; imgCnt++) { 
        vector<uint8_t> &img = imgs[imgCnt];
        uint8_t &label = labels[imgCnt];
        fread(&img[0], imgsize, 1, fp);
        fread(&label, 1, 1, flabel);
    }
    cout << fixed << setprecision(12);

    Model model(0.000001);
    
    for(int t = 0; ; t++) {
        for(int i = 10; i < 2000; i++) {
            model.train(imgs[i], labels[i]);
            //model.train(imgs[0], labels[0]);
        }
        cout << "-------------------------------\n";
        double avg_err = 0;
        for(int i = 0; i < 10; i++) {
            avg_err += model.Er(imgs[i], labels[i]);
        }
        cout << "(" << t << ") Avg Error: " << avg_err << endl;
    }
    cout << "train finish" << endl;
    
}






