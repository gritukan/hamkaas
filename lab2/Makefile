NVCC = nvcc
FLAGS = --std=c++17 -G -g -O2

SRCS = 00.cu 01.cu 02.cu 03.cu 04.cu 05.cu 06.cu 07.cu 08.cu

TARGETS = $(SRCS:.cu=)

all: $(TARGETS)

$(TARGETS): %: %.cu
	$(NVCC) $(FLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
