SRCS = allocator.cpp interface.cpp model.cpp node.cpp parser.cpp tensor.cpp

so: $(SRCS)
	nvcc --compiler-options -fPIC -shared -g -O2 -o libhamkaas.so $(SRCS)
