SRCS = interface.cpp model.cpp node.cpp parser.cpp tensor.cpp

so: $(SRCS)
	nvcc --compiler-options -fPIC -shared -g -o libhamkaas.so $(SRCS)

