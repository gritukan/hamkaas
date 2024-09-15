SRCS = interface.cpp

so: $(SRCS)
	nvcc --compiler-options -fPIC -shared -o libhamkaas.so $(SRCS)

