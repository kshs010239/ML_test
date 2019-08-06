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
typedef pair<double, double> PairF;




double e = exp(1);

double L(double x) {
    return 1.0 / (1 + exp(-x));
}


double Random( double Max = 0.01, double Min = -0.01) {
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

double D(double x, double e) {
    return e * pow(x, e - 1);
}

vector<int> choose(int n, int Max, int Min = 0) {
   vector<int> ret;
   if(Max - Min < n) return vector<int>();

   while(ret.size() < n) {
        int r = randint(Max, Min);
        int dep = 0;
        for(auto &it: ret)
            if(it == r) {
                dep = 1;
                break;
            }
        if(dep)
            continue;
        ret.push_back(r);
   }
   return ret;
}


struct Model {

    double alpha;

  
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
            cout << "LC show:" << endl;
            for(int i = 0; i < 2 /*W[0].size()*/; i++) {
                for(int j = 0; j < W.size(); j++) {
                    char out = ' ';
                    double val = W[j][i] > 0 ? W[j][i] : -W[j][i];
                    val = W[j][i];
                    if(val > 0.000) out = '.';
                    if(val > 0.008) out = '*';
                    cout << out << (j % 28 == 27 ? "\n" : "");
                }
                cout << "\n--------------------------------\n";
            }
        }
    };

    struct NFLC { //  Not fully linear connect
        struct Node {
            int x;
            double w;
            Node(int x = 0, double w = 0): x(x), w(w) {}
        };
        vector<vector<Node> > W;
        vector<vector<Node> > BW;
        double alpha;
        int outSize;
        NFLC(double alpha, int in, int out): alpha(alpha), W(in), BW(in), outSize(out) {
            for(int i = 0; i < W.size(); i++) {
                int r = randint(sqrt(outSize));
                int cn = max(3, r);
                vector<int> v = choose(cn, out, 0); // choose cn from out to 0.
                for(int j = 0; j < cn; j++) {
                    int r = Random();
                    W[i].push_back(Node(v[j], r));
                    BW[v[j]].push_back(Node(i, r));
                }
            }
        }
        void forward(Layer& L1, Layer& L2) {
            setz(L2);

            /*for(int j = 0; j < L2.size(); j++)
                for(int i = 0; i < L1.size(); i++)
                    L2[j] += L1[i] * W[i][j];
            */
            for(int i = 0; i < W.size(); i++)
                for(auto &it: W[i])
                    L2[it.x] += L1[i] * it.w;
            
        }
        void backward(Layer& L1, Layer& L2) {

            for(int i = 0; i < W.size(); i++)
                for(auto &it: W[i])
                    it.w -= alpha * L2[it.x] * L1[i]; 
            
            setz(L1);
            for(int i = 0; i < W.size(); i++)
                for(auto &it: W[i])
                    L1[i] += L2[it.x] * it.w;
        }

    };

    struct AndC {
        vector<Pair> from;
        vector<PairF> weight;
        int inSize;
        double alpha;
        AndC(double alpha, int in, int out): alpha(alpha), inSize(in), from(out - in), weight(out - in) {
            for(int i = 0; i < from.size(); i++) {
                from[i] = Pair(randint(in), randint(in)); 
                weight[i] = PairF(Random(1, -1), Random(1, -1));
            }
        }
        void forward (Layer& L1, Layer& L2) {
            setz(L2);

            for(int i = 0; i < from.size(); i++) {
                Pair& pr = from[i];
                PairF& W = weight[i];
                L2[i + inSize] = pow(L1[pr.first], L(W.first)) * pow(L1[pr.second], L(W.second));
            }
            for(int i = 0; i < inSize; i++)
                L2[i] = L1[i];
            DB(L1);
            DB(L2);
        }
        void backward( Layer& L1, Layer& L2) {

            for(int i = 0; i < from.size(); i++) {
                Pair& pr = from[i];
                PairF& W = weight[i];
                double delta = alpha * L2[i + inSize] * pow(L1[pr.first], L(W.first)) * pow(L1[pr.second], L(W.second));
                W.first  -= delta * (1 - L(W.first) ) * L(W.first)  * log(L1[pr.first ]);
                W.second -= delta * (1 - L(W.second)) * L(W.second) * log(L1[pr.second]);
            }

            Layer L1t = L1;
            setz(L1);
            for(int i = 0; i < inSize; i++)
                L1[i] += L2[i];
            for(int i = 0; i < from.size(); i++) {
                Pair& pr = from[i];
                PairF& W = weight[i];
                double tmp1 = L1[pr.first];
                double tmp2 = L1[pr.second];
                L1[pr.first ] += D(L1t[pr.first ], L(W.first) ) * pow(L1t[pr.second], L(W.second)) * L2[i + inSize];
                L1[pr.second] += D(L1t[pr.second], L(W.second)) * pow(L1t[pr.first ], L(W.first) ) * L2[i + inSize];
                //cout << tmp1 << " " << L1[pr.first] << " | " << tmp2 << " " << L1[pr.second] << endl;
            }
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

    Layer 
        imgdata, 
        hd, hd2, hd3, hd4, hd5, 
        hd6, hd7, hd8,
        out, out2;

    LC lc1, lc2, lc3;
    LogC logc;
    AndC andc, andc2, andc3, andc4;

    
    Model(double alpha = 0.00002, int sz1 = 200, int szAnd = 200, int sz2 = 40, int sz3 = 10): 
        alpha(alpha), imgdata(imgsize), 
        hd(sz1), hd2(sz1), hd3(sz1 + szAnd), hd4(sz1 + 2*szAnd), hd5(sz1 + 3*szAnd),
        hd6(sz2), hd7(sz2), hd8(sz2 + szAnd), 
        out(sz3), out2(sz3), 
        lc1(   alpha, imgsize, sz1), 
        andc(  alpha, sz1,           sz1 +   szAnd),
        andc2( alpha, sz1 +   szAnd, sz1 + 2*szAnd),
        andc3( alpha, sz1 + 2*szAnd, sz1 + 3*szAnd),
        lc2(   alpha, sz1 + 3*szAnd, sz2),
        andc4( alpha, sz2,           sz2 +   szAnd), 
        lc3(   alpha, sz2 + szAnd,   sz3)
        {}



    void _predict (Image& img) {
        for(int i = 0; i < imgsize; i++) 
            imgdata[i] = img[i];
        
        lc1.forward(imgdata, hd);
        logc.forward(hd, hd2);
        andc.forward(hd2, hd3); 
        andc2.forward(hd3, hd4);
        andc3.forward(hd4, hd5);
        lc2.forward(hd5, hd6);
        logc.forward(hd6, hd7);
        andc4.forward(hd7, hd8);
        lc3.forward(hd8, out);
        logc.forward(out, out2);
    }
    void _backward(Image& img, uint8_t label) {
        for(int i = 0; i < out2.size(); i++)
            if(i == label) {
                out2[i] = 8 * (out2[i] - 1);
            }
        logc.backward(out, out2);
        lc3.backward(hd8, out);
        andc4.backward(hd7, hd8);
        logc.backward(hd6, hd7);
        lc2.backward(hd5, hd6);
        andc3.backward(hd4, hd5);
        andc2.backward(hd3, hd4);
        andc.backward(hd2, hd3);
        logc.backward(hd, hd2);
        lc1.backward(imgdata, hd); 
    }
    void predict (Image& img) {
        _predict(img);
    }
    double Er(Image& img, uint8_t label) {
        _predict(img);
        
        
        cout << "(" << (int)label << ") ";
        for(int i = 0; i < 10; i++)
            cout << i << ": " << out2[i] << "\t";
        cout << endl;
        

        double ret = 0;
        for(int i = 0; i < 10; i++) {
            double er = (i == label ? 1 - out2[i]: out2[i]);
            ret += er * er;
        }
        return (ret);
    }
    int T(Image& img, uint8_t label) {
        _predict(img);
        int ret = 0;
        for(int i = 0; i < 10; i++) {
            if(out2[ret] < out2[i])
                ret = i;
        }
        return ret == label;
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

void showtime() {
   time_t now = time(0);
   tm *ltm = localtime(&now);

   // print various components of tm structure.
   cout 
        << "Time: "
        << setw(2) << setfill('0')
        << ltm->tm_hour << ":"
        << setw(2) << setfill('0')
        << ltm->tm_min << ":"
        << setw(2) << setfill('0')
        << ltm->tm_sec << endl; 
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
    
    cout << "Dataset size: " << numberOfImages << endl << endl;
    

    vector<vector<uint8_t> > imgs(numberOfImages, vector<uint8_t>(imgsize));
    vector<uint8_t> labels(numberOfLabels);

    for(int imgCnt = 0; imgCnt < numberOfImages; imgCnt++) { 
        vector<uint8_t> &img = imgs[imgCnt];
        uint8_t &label = labels[imgCnt];
        fread(&img[0], imgsize, 1, fp);
        fread(&label, 1, 1, flabel);
    }
    cout << fixed << setprecision(5);

    Model model;
    
    for(int t = 0; ; t++) {
        cout << "-------------------------------\n";
		cout << "(" << t << ") ";
		showtime();
        for(int i = 1000; i < 2000; i++) {
        //for(int i = 1000; i < 2000; i++) {
            model.train(imgs[i], labels[i]);
        }
        double avg_err = 0;
        int TP = 0;
        for(int i = 0; i < 1000; i++) {
            avg_err += model.Er(imgs[i], labels[i]);
            TP += model.T(imgs[i], labels[i]);
        }
        cout << "Avg Error: " << avg_err << endl;
        cout << "TP: " << TP << endl;

    }
    cout << "train finish" << endl;
    
}






