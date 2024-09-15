SRCS = allocator.cpp bootstrap.cpp device.cpp helpers.cpp interface.cpp kernels.cu model.cpp node.cpp parser.cpp tensor.cpp

so: $(SRCS)
	nvcc --compiler-options -fPIC -shared -lcublas -g -o libhamkaas.so $(SRCS)
