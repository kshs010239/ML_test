#include "Model.h"
#include <functional>
#include <iostream>
#include <iomanip>

typedef uint8_t Label;

int ReadInt(FILE* p) {
    int ret = 0;
    uint8_t buf[10];
    fread(buf, 1, 4, p);
    for(int i = 0; i < 4; i++)
        ret = ret * 256 + buf[i];
    return ret;
}


vector<Data> ReadImage(FILE* fp) {
	int magic_num = ReadInt(fp);
	int num_images = ReadInt(fp);
	int width = ReadInt(fp);
	int height = ReadInt(fp); 

	int imgsize = width * height;
	vector<Data> images(num_images, Data(imgsize));
    for(Data& image: images) {
        vector<uint8_t> raw_image(imgsize);
        fread(raw_image.data(), imgsize, 1, fp);
        for(int i = 0; i < imgsize; i++) {
            image[i] = (double)((int)raw_image[i] + 0.5) / 256;
            assert(image[i] < 1 and image[i] > 0);
        }

    }

	return images;
}

vector<Label> ReadLabel(FILE* fp) {
	int magic_num = ReadInt(fp);
    int num_labels = ReadInt(fp);
	
    vector<Label> labels(num_labels);
    for(Label& label: labels)
        fread(&label, 1, 1, fp);

    return labels;
}

void errdump(const char *s) {
    std::cout << s << std::endl;
    exit(0);
}

void InvalidArgument(int no = 0) {
    std::cout << no << ": ";
    errdump("invalid argument");
}

FILE* Critical_fopen(const char *name, const char* mode) {
    if(name == NULL)
        InvalidArgument(1);
    FILE* ret = fopen(name, mode);
    if(ret == NULL)
        InvalidArgument(2);
    return ret;
}

int main(int argc, char *argv[]) {
    cout << std::fixed << std::setprecision(3);

    if(argc < 5)
        InvalidArgument();
    FILE* train_image_file = Critical_fopen(argv[1], "r");
    FILE* train_label_file = Critical_fopen(argv[2], "r");
    FILE* test_image_file  = Critical_fopen(argv[3], "r");
    FILE* test_label_file  = Critical_fopen(argv[4], "r");
    

    vector<Data> train_images = ReadImage(train_image_file);
    vector<Label> train_labels = ReadLabel(train_label_file);
    vector<Data> test_images = ReadImage(test_image_file);
    vector<Label> test_labels = ReadLabel(test_label_file);
    int train_size = train_images.size();


	Model<Label> model;
    std::function<double()> randomInit  = std::bind(Random::Random, 0.1, -0.1);
    std::function<double()> randomInit2  = std::bind(Random::Random, 0.1, -0.1);
    std::function<double()> randomInit3 = std::bind(Random::Random, 1, -1);
    model.AddLayer(new FullConnect<Multipy> (784, 10, randomInit3, 0.01));
    model.AddLayer(new FullConnect<Multipy> (10, 10, randomInit3, 0.01));
    //model.AddLayer(new Activation<>    (150));
    //model.AddLayer(new FullConnect<Linear> (150, 40, randomInit2, 0.00001));
    //model.AddLayer(new Activation<Sigmoid> (40));
    //model.AddLayer(new FullConnect<Multipy>(40, 10, randomInit3, 1));
    /*
    //model.AddLayer(new Activation<Sigmoid>(10));
    model.AddLayer(new FullConnect<Linear> (60,  40, randomInit3, 0.0003));
    model.AddLayer(new Activation<ReLU>    (40));
    model.AddLayer(new FullConnect<Linear> (40,  30, randomInit3, 0.0002));
    model.AddLayer(new Activation<Sigmoid> (30));
    model.AddLayer(new FullConnect<Multipy>(30,  20, randomInit3, 1));
    model.AddLayer(new FullConnect<Linear> (20,  10, randomInit3, 0.0001));
    model.AddLayer(new Activation<Sigmoid> (10));
    */


    for(int t = 0; ; t++) {
        double total_err = 0;
        int TP = 0;
        /*for(int i = 0; i < num_sample; i++) {
            total_err += Loss(model, train_images[i], train_labels[i]);
            //cout << "(" << (int)train_labels[i] << "): " 
            //     << model.PredictResult(train_images[i]) << endl;
        }*/
        int num_trained = 1;
        for(int i = 0; i < num_trained; i++) {
            int idx = Random::Randint(train_size);
            model.Train(train_images[idx], train_labels[idx]);
            const auto& res = model.getResult();
            total_err += Loss(res, train_images[i], train_labels[i]);
            if(model._Predict(res) == train_labels[idx]) 
                TP += 1;
        }
        cout << "avg err: " << total_err / num_trained << endl;
        cout << "TP: " << (double)TP / num_trained * 100 << endl;
        cout << "------------------------------\n";
    }

	return 0;
}
