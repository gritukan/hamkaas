SRCS = allocator.cpp bootstrap.cpp device.cpp helpers.cpp interface.cpp kernels.cu model.cpp node.cpp parser.cpp tensor.cpp

FLAGS = --compiler-options -fPIC -shared -lcublas -O2 -g -o libhamkaas.so

debug: $(SRCS)
	nvcc $(FLAGS) -G $(SRCS)

release: $(SRCS)
	nvcc $(FLAGS) -Xptxas -O3,-v $(SRCS)
