FLAGS = --std=c++17 -G -g -O2

01: 01.cpp
	nvcc $(FLAGS) -o 01 01.cpp

02: 02.cu
	nvcc $(FLAGS) -lcublas -o 02 02.cu

03: 02.cu
	nvcc $(FLAGS) -DUSE_CUBLAS -lcublas -o 03 02.cu

04: 04.cu
	nvcc $(FLAGS) -lcublas -lcudnn -o 04 04.cu
