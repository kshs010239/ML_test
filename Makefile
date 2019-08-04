

all: show.cpp
	g++ -std=c++14 -o main show.cpp 
test: main
	./main train-images.idx3-ubyte  train-labels.idx1-ubyte

test2: 
	g++ -std=c++14 -o main2 show2.cpp 
	./main2 train-images.idx3-ubyte  train-labels.idx1-ubyte
    

