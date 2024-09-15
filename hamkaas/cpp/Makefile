SRCS = allocator.cpp bootstrap.cpp cudnn_optimizer.cpp device.cpp helpers.cpp interface.cpp kernels.cu model.cpp node.cpp parser.cpp tensor.cpp

FLAGS = --compiler-options -fPIC -shared -lcublas -O2 -g -o libhamkaas.so

CUDNN_FLAGS = -lcudnn -DUSE_CUDNN

debug: $(SRCS)
	nvcc $(FLAGS) -G $(SRCS)

release: $(SRCS)
	nvcc $(FLAGS) -Xptxas -O3,-v $(SRCS)

debug-cudnn: $(SRCS)
	nvcc $(FLAGS) $(CUDNN_FLAGS) -G $(SRCS)

release-cudnn: $(SRCS)
	nvcc $(FLAGS) $(CUDNN_FLAGS) -Xptxas -O3,-v $(SRCS)
