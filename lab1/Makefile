NVCC = nvcc
FLAGS = --std=c++17

SRCS = 00.cu 01.cu 02.cu 03.cu 04.cu 05.cu

TARGETS = $(SRCS:.cu=)

all: $(TARGETS)

$(TARGETS): %: %.cu
	$(NVCC) $(FLAGS) -o $@ $<

00-test: 00
	./00

01-test: 01
	./01

02-test: 02
	./02

03-test: 03
	./03

04-test: 04
	./04

05-test: 05
	./05

test-all: 00-test 01-test 02-test 03-test 04-test 05-test

clean:
	rm -f $(TARGETS)
