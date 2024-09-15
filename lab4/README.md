In the previous lab, you wrote optimized code for the specific neural network for digit classification. In this lab, you will start working on HamKaas compiler that will execute arbitrary neural networks.

Note, that for this and the next lab the code is located in hamkaas directory but not in the lab directory. That is because in lab6 you will need to use code you wrote in this lab.

The hamkaas directory contains the source code of the compiler we are going to build. It is not complete and not working now and your task is to implement the missing parts.

## Architecture

Before we start, let's discuss the top-level architecture of the HamKaas compiler. HamKaas compiler consists of the two main components: the frontend and the backend.

The frontend is a Python library that is used to define the neural network. It provides a PyTorch-like API: there are classes for tensors and different operations on them. The frontend is responsible for converting the Python-defined neural network into some intermediate representation which is the script in a special language. We will look at this language later.

The backend is a C++ program that parses the script produced by frontend and executes in on the GPU and CPU. It is also responsible for all the optimizations. The backend is distributed as a [shared library](https://en.wikipedia.org/wiki/Shared_library) that is loaded by frontend.

## Frontend and Script Language

Before we start to actually work with the code, let's look at the script language that is used to represent the neural network. Instead of long explanations, let's look at the example of the script that represents the neural network from the previous lab. Open the file `hamkaas/python/mnist.py`. Look how the neural network is defined and then run the script. You should get the following output.

```
$1 = InputTensor(input, float32, [128, 28, 28]);
$2 = ReshapeNode($1, [128, 784]);
$3 = ConstantTensor(constant_0, float32, [784, 1000]);
$4 = MulNode($2, $3);
$5 = ConstantTensor(constant_1, float32, [1, 1000]);
$6 = SumNode($4, $5);
$7 = ReLUNode($6);
$8 = ConstantTensor(constant_2, float32, [1000, 10]);
$9 = MulNode($7, $8);
$10 = ConstantTensor(constant_3, float32, [1, 10]);
$11 = SumNode($9, $10);
result = $11;
```

This is how the neural network is represented in the script. Formally, the script is a sequence of statements with each statement defining the node. Each node has a integer identifier (e.g., `$1`, `$2`, etc.). There is a number of operators that use other nodes as inputs. For example `$6 = SumNode($4, $5);` means that the node `$6` is a scalar sum of nodes `$4` and `$5`. The last statement defines the output of the neural network.

For simplicity, we assume that nodes in the script are [topologically ordered](https://en.wikipedia.org/wiki/Topological_sorting). That is, if the node $a$ is used as an input for the node $b$, then $a$ is defined before $b$ in the script.

This language is similar to analogues in real compilers. For example, in PyTorch the [TorchScript](https://pytorch.org/docs/stable/jit.html) is used. You can look at what it looks like [here](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html#basics-of-torchscript). Similar, isn't it? The same is used in other compilers as well. For example, JAX uses [XLA](https://www.tensorflow.org/xla) compiler that uses [HLO](https://openxla.org/stablehlo) as an intermediate representation (example [here](https://openxla.org/stablehlo/spec#programs)).

Note, that traversal result contains not only the script. It also contains the list of the input tensors, the information about the output tensors and values of the constant tensors. All of this information will be passed to the backend.

## HamKaas Source Structure

Before we start working with the code let's look at the structure of the HamKaas source code.

The `hamkaas` directory contains the `python` and `cpp` directories. The `python` directory contains the frontend code and the `cpp` directory contains the backend code.

In the `python` directory `hamkaas.py` file defines the HamKaas frontend. `tests` directory contains the compiler tests.

The `cpp` directory has the following files:
- `allocator.h, allocator.cpp, allocator_ut.cpp` - memory allocator that will be used to determine the memory layout of the computation graph.
- `bootstrap.h, bootstrap.cpp` - contains the entities that are resued between different models like cuBLAS and cuDNN handles.
- `cudnn_optimizer.h, cudnn_optimizer.cpp` - contains the code that fuses the nodes by using cuDNN.
- `device.h, device.cpp` - contains the abstraction for the device-specific operations.
- `error.h` - contains some macros for error handling.
- `helpers.h, helpers.cpp` - contains some helpers.
- `interface.h, interface.cpp` - contains the interface of the shared library that is used for the communication between the frontend and the backend.
- `kernels.h, kernels.cu` - contains the CUDA kernels that are used for node execution.
- `model.h, model.cpp` - contains the code that compiles and executes the model.
- `node.h, node.cpp` - contains the code that represents the nodes of the computation graph. It has a class for every node type (e.g., `MulNode`, `SumNode`, etc.) that defines the node behavior.
- `parser.h, parser.cpp` - contains the code that parses the script and converts it into the tree of nodes.
- `tensor.h, tensor.cpp` - contains the `TTensorMeta` class that represents the tensor type and shape.

## 01: Dynamic Linkage

Let's start working on the compiler. Right now, the frontend is not able to communicate with the backend because shared library loader is not implemented. Let's fix that!

If you are not familiar with shared libraries, you can read about them [here](https://en.wikipedia.org/wiki/Shared_library), for example.

To load the shared library in Python code we will use the `ctypes` module. You can read about it [here](https://docs.python.org/3/library/ctypes.html).

The interface of the shared library is defined in the file `hamkaas/cpp/interface.h`. At first, you need to support the functions `HamKaasInverseElements` and `HamKaasFreeErrorMessage` in Python code.

`HamKaasInverseElements` is a toy function that takes `in` and `out` arrays of floats and performs `out[i] = 1.0 / in[i]` for each element. This function returns a pointer to the error message. If the pointer is null, then the function was executed successfully. Otherwise (in case of division by zero), the pointer points to the error message. When the error message is processed by the Python code, the memory should be freed using the `HamKaasFreeErrorMessage` function.

Open the `hamkaas.py` file and find the class `HamKaasPlugin`. You need to implement the `inverse_elements` function. To do so, you need to define the function signatures in the `__init__` function. Then you need to call the defined functions from the `inverse_elements` function.

To define the function signatures, you need to set the `argtypes` and `restype` attributes of the function. You can see the example of how to do it [here](https://docs.python.org/3/library/ctypes.html#return-types). Hint: the type for the returning string is `ctypes.POINTER(ctypes.c_ubyte)` and the type for a float array is `ctypes.POINTER(ctypes.c_float)`.

When implementing the `inverse_elements` function, you will need to convert torch tensors to the float arrays. To do so, you will need to use `.data_ptr()` method of the tensor and `ctypes.cast` function. Note, that the tensor is not always contiguous, so in order to use `.data_ptr()` method to get the pointer to the data, you need to call `.contiguous()` method on the tensor.

After you implemented the `inverse_elements` function, you can run the test. At first, run `make debug` in the `hamkaas/cpp` directory to build the shared library. Then run the test in the `hamkaas/python/tests` directory by running `pytest -sv -k TestInverseElements`. If all tests are passed, good job! You managed to call the C++ function from Python code.

Now, you need to implement the rest of the functions from the `interface.h` file in the `HamKaasPlugin` class.

Let's start with `HamKaasInitialize` and `HamKaasFinalize`. The `HamKaasInitialize` initializes the internal structures of the compiler and returns some opaque handler that should be passed to other functions. The `HamKaasFinalize` destroys these structures. You can look at their implementation in the `interface.cpp` file: `HamKaasInitialize` just creates an instance of `TBootstrap` and `HamKaasFinalize` removes it.

In the plugin, the `HamKaasInitiaize` should be called in the `__init__` function and the handle should be stored in plugin. The `HamKaasFinalize` should be called in the `__del__` function. When you are ready, you can test if everything works by creating the file `test.py` in `hamkaas/python` directory and running something like this:

```python
import hamkaas

plugin = hamkaas.HamKaasPlugin("../cpp/libhamkaas.so")
```

You can add some print statements in the C++ code to check if everything works as expected.

Next, you need to implement `compile_model` of the `HamKaasPlugin` class that calls `HamKaasCompileModel` function of the shared library. After that, add the code to the `HamKaasModel` class that represents model. The `evaluate` method should call `HamKaasEvaluateModel` and the `HamKaasFreeModel` should be called from `__del__`. Take a look at the signatures and try to understand what each function does.

Unfortunately, the compiler is not ready yet to run anything, so you cannot test your implementation yet.

## 02: Memory Allocator and Control Flow

Just like in the previous lab, in HamKaas we will take an advantage of the model being static and will allocate all the required memory in advance. This will allow us to avoid memory allocations during the inference and will make code simpler and faster because all addresses will be known in advance.

HamKaas uses the following memory model for the graph execution. Each node has two kind of the allocated memory: the output memory and the buffer memory. The output memory is the memory where the result of the node is stored. The buffer memory is the memory that is used for the intermediate computations of the node. By default, for every node the output memory size is determined by the shape of its output tensor. However, some nodes are "transparent" and return the pointer to the input memory. For example, the `ReshapeNode` is transparent since it just changes the tensor metadata without changing the data. Another example of the transparent node is the `ReplaceSliceNode` that changes the input tensor but returns the pointer to the input so does not have the output memory.

Memory allocator is closely related to the evaluation control flow. Consider some two nodes $A$ and $B$ of the model. If $A$ and $B$ are never executed concurrently then the memory allocator can take advantage of this fact and reuse the buffer memory of $A$ for the buffer of $B$. Generally, the memory allocation should preserve two rules:
- Buffer memory of the node should be available during the execution of the node.
- Ouptut memory of the node should be available during the execution of the nodes that use the output of the node.

Note that these rules imply some memory-performance trade-offs. Greater buffer memory reuse leads to the smaller memory consumption but can reduce the parallelism of the execution since two nodes sharing the buffer or the output memory cannot be executed concurrently. In HamKaas we execute all the nodes sequentially since it is simpler, so we can take advantage of the buffer memory reuse.

Let's start implementing memory allocator. Open the file `allocator.h` in the `hamkaas/cpp` directory. You need to implement the `TAllocator` class. Note, that this class does not really allocate the memory but just computes the memory layout of the nodes.

The intended usage of the allocator is following: during the model compilation, the allocator is created and the model graph is traversed. When buffer or output memory is required, compiler calls `Allocate` method and gets the offset of the memory in future memory block. When memory is no longer needed (e.g., the node is executed), the compiler calls `Free` method to free the memory. After the graph traversal is completed, `GetWorkingSetSize` is called to get the amount of memory required for the memory execution. After that, the real memory block of size `GetWorkingSetSize` is allocated at address `base` and each node receives the memory address `base + offset` where `offset` is the value returned by the `Allocate` method.

Implement the `TAllocator` class. If you want to keep things simple, you can implement the allocator that just allocates the memory sequentially. If you want to have more challedge, you can implement some advanced memory allocation strategy that reuses memory after `Free` calls. In order to avoid thinking about the memory alignment, all the allocations should be aligned to 256 bytes (i.e. for all allocations `offset % 256 = 0`). `cudaMalloc` also aligns all allocations by 256 bytes, so after adding `base` addresses will be still aligned.

When you completed the allocator, you can run simple unittest by executing `make test-allocator` in `hamkaas/cpp` directory. If you see `All tests passed!`, good job! You can move to the next step.

After the allocator is completed, you can use it to allocate buffer and output memory for the nodes. Open the `model.cpp`, you will implement `TModel::AllocateMemory()` method.

`EvaluationOrder_` is the vector that contains all non-input nodes in the topological order. During the inference, the nodes are executed in this order. Remember, that HamKaas never executes two nodes concurrently. `InputNodes_` is the vector that contains all input nodes. The following methods of the `TNodeBase` are needed for the memory allocations:

- `GetBufferSize()` returns the amount of buffer memory required for the node.
- `GetOutputSize()` returns the amount of output memory required for the node.
- `SetBuffer(char* buffer)`, `SetOutput(char* output)` set the buffers for the node.
- `GetOutputOwner()` returns the node that allocated the memory for the output of this node. Typically, it is the node itself (so, `node->GetOutputOwner() == node`) but for the aforementioned transparent nodes it can be other node. For example if there is a `MulNode` and `ReshapeNode` that takes the result of the `MulNode` as an input, then the `ReshapeNode` should return the `MulNode` as the output owner. This method is used to understand when some output memory can be freed.

To allocate the real memory for the nodes, you need to call `Device_->DeviceMalloc` method. `Device` is the abstraction used in HamKaas to encapsulate some device-specific operations. Right now, there is only one device - `CpuDevice` defined in `device.h` that is used in case of the CPU execution. You will implement `CudaDevice` for the GPU execution later.

When you are ready, compile the `libhamkaas.so` library by running `make debug` in the `hamkaas/cpp` directory. Then you can test your implementation by running the tests in the `hamkaas/python/tests` directory by running `pytest -sv -k TestCompilerCpu`. This will run all the HamKaas tests with models executing on the CPU. If all tests are passed, good job! Your compiler is now working!
