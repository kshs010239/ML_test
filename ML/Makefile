

all: main.cpp
	g++ -std=c++11 main.cpp -o main

clean: main
	rm main

test: main
	./main ../data/train-images.idx3-ubyte  ../data/train-labels.idx1-ubyte ../data/train-images.idx3-ubyte  ../data/train-labels.idx1-ubyte

gdb: main
	gdb -q --args main ../data/train-images.idx3-ubyte  ../data/train-labels.idx1-ubyte ../data/train-images.idx3-ubyte  ../data/train-labels.idx1-ubyte

