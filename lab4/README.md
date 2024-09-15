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
$4 = MatMulNode($2, $3);
$5 = ConstantTensor(constant_1, float32, [1, 1000]);
$6 = SumNode($4, $5);
$7 = ReLUNode($6);
$8 = ConstantTensor(constant_2, float32, [1000, 10]);
$9 = MatMulNode($7, $8);
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
- `node.h, node.cpp` - contains the code that represents the nodes of the computation graph. It has a class for every node type (e.g., `MatMulNode`, `SumNode`, etc.) that defines the node behavior.
- `parser.h, parser.cpp` - contains the code that parses the script and converts it into the tree of nodes.
- `tensor.h, tensor.cpp` - contains the `TTensorMeta` class that represents the tensor type and shape.

## HamKaas Script Specification

Since you will implement different nodes in the backend, you will need a formal specification of the script.

### Tensors

HamKaas operates with the tensor with tensor being a multi-dimensional array. Each tensor is represented by type and shape.

The type can be either a `float32` which is represented by the `float` type in C++ or a `int64` which is represented by the `int64_t` type in C++.

The shape is a list of integers that represent the size of each dimension. For example, the shape `[128, 28, 28]` means that the tensor is a 3D tensor with the size of 128x28x28. HamKaas supports tensors with 1, 2 or 3 dimensions. HamKaas does not support empty tensors, so each dimension should have at least one element.

### Nodes

#### InputTensor
* Syntax: `InputTensor(name, type, shape)`
* Description: describes the input tensor with given type and shape. During the evaluation, the input tensor with the given name should be passed to the model.
* Usage: used to define the input of the model.
* Inputs: none
* Output: the tensor with the given type and shape.
#### ConstantTensor
* Syntax: `ConstantTensor(name, type, shape)`
* Description: describes the tensor with given type and shape that is constant during the model execution. During the compilation, the tensor with the given name should be passed to the model.
* Usage: used to define the model parameters, for example, weights.
* Inputs: none
* Output: the tensor with the given type and shape.
#### BufferTensor
* Syntax: `BufferTensor(name, type, shape)`
* Description: describes the tensor with given type and shape that is allocated during the model compilation and is not changed between the model executions.
* Usage: data that is passed between different model runs, for example, caches.
* Inputs: none
* Output: the tensor with the given type and shape.
#### SumNode
* Syntax: `SumNode(lhs, rhs)`
* Description: describes the node that computes the element-wise sum of two tensors. This node supports broadcasting, that is, if the shapes of the tensors are different, the `rhs` tensor is broadcasted to the shape of the `lhs` tensor. You can read more about broadcasting [here](https://numpy.org/doc/stable/user/basics.broadcasting.html).
* Inputs: two tensors of type `float32` with the same number of dimensions. For each axis, the size of the axis should be the same or the size of the axis in `rhs` should be 1.
* Output: the tensor with the same type and shape as `lhs`.
#### HadamardProductNode
* Syntax: `HadamardProductNode(lhs, rhs)`
* Description: describes the node that computes the element-wise product of two tensors. This node supports broadcasting, that is, if the shapes of the tensors are different, the `rhs` tensor is broadcasted to the shape of the `lhs` tensor.
* Inputs: two tensors of type `float32` with the same number of dimensions. For each axis, the size of the axis should be the same or the size of the axis in `rhs` should be 1.
* Output: the tensor with the same type and shape as `lhs`.
#### ReLUNode
* Syntax: `ReLUNode(input)`
* Description: describes the node that computes the element-wise ReLU of the input tensor. The ReLU function is defined as `ReLU(x) = max(0, x)`.
* Inputs: the tensor of type `float32`.
* Output: the tensor with the same type and shape as `input`.
#### SiLUNode
* Syntax: `SiLUNode(input)`
* Description: describes the node that computes the element-wise SiLU of the input tensor. The SiLU function is defined as `SiLU(x) = x / (1 + exp(-x))`.
* Inputs: the tensor of type `float32`.
* Output: the tensor with the same type and shape as `input`.
#### MatMulNode
* Syntax: `MatMulNode(lhs, rhs)`
* Description: performs the matrix multiplication. Works in three modes:
    - If both `lhs` and `rhs` are 2D tensors, then the node computes the matrix multiplication of the two tensors.
    - If `lhs` is a 1D tensor and `rhs` is a 2D tensor, then the node computes the
    vector-matrix multiplication.
    - If `lhs` is a 3D tensor and `rhs` is a 3D tensor, then the node computes the
    batched matrix multiplication, that is, interprets `lhs` and `rhs` as the array of matrices and computes the matrix multiplication for each pair of matrices.
* Inputs: two tensors of type `float32`. The following shapes are allowed:
    - For matrix multiplication `[m, n]` and `[n, k]`.
    - For vector-matrix multiplication `[n]` and `[n, k]`.
    - For batched matrix multiplication `[b, m, n]` and `[b, n, k]`.
* Output: the tensor of type `float32`. The shape of the output tensor is determined by the shapes of the input tensors.
    - For matrix multiplication `[m, k]`.
    - For vector-matrix multiplication `[k]`.
    - For batched matrix multiplication `[b, m, k]`.
#### SliceNode
* Syntax: `SliceNode(input, begin, end)`
* Description: returns a slice of the input tensor over the first axis. The slice is defined by the constant `begin` and `end` parameters.
* Inputs: the tensor of type `float32` of type `float32` and shape `[n, ...]` and two integers `begin` and `end` such that `0 <= begin <= end <= n`.
* Output: the tensor of type `float32` and shape `[end - begin, ...]` built from the tensor elements `input[begin], input[begin + 1], ..., input[end - 1]`.
#### ReshapeNode
* Syntax: `ReshapeNode(input, shape)`
* Description: changes the shape of the input tensor. The number of elements in the input tensor should be the same as the number of elements in the output tensor. Semantics is the similar to [numpy.reshape](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html) function.
* Inputs: the tensor of type `float32` and the list of integers `shape` that represents the new shape of the output tensor.
* Output: the tensor of type `float32` and the shape `shape`.
#### PermuteNode
* Syntax: `PermuteNode(input, permutation)`
* Description: changes the order of the dimensions of the input tensor. The number of elements in the input tensor should be the same as the number of elements in the output tensor. Semantics is the similar to [numpy.transpose](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html) function.
* Inputs: the tensor of type `float32` with $d$ axis and and the permutation of the $d$ integers that represents the new order of the dimensions.
* Output: the tensor of type `float32` with the same shape as the input tensor but with the dimensions permuted.
#### ReplaceSliceNode
* Syntax: `ReplaceSliceNode(input, replacement, begin, end)`
* Description: replaces the slice of the input tensor over the first axis with the replacement tensor. The slice is defined by the constant `begin` and `end` parameters. Note that this operation does not create a new tensor but changes the input tensor in-place.
* Usage: the intended usage is to update buffer tensors.
* Inputs: the tensor of type `float32` of type `float32` and shape `[n, ...]`, the tensor of type `float32` and shape `[k, ...]` with matching shapes except the first axis, and two 1D tensors of type `int64` `begin` and `end` having the size 1.
* Output: the tensor of type `float32` and shape `[n, ...]` where the elements `input[begin], input[begin + 1], ..., input[end - 1]` are replaced with the elements of the `replacement` tensor. If `end - begin != k` is observed during runtime, the behavior is undefined.

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

HamKaas uses the following memory model for the graph execution. Each node has three kinds of the allocated memory: the constant memory, the output memory and the buffer memory.

The constant memory stores the node-related data that does not change during all the model execution lifetime (e.g., storage of the constant node or some auxiliary data like tensor shapes).

The output memory is the memory where the result of the node is stored. The buffer memory is the memory that is used for the intermediate computations of the node. By default, for every node the output memory size is determined by the shape of its output tensor. However, some nodes are "transparent" and return the pointer to the input memory. For example, the `ReshapeNode` is transparent since it just changes the tensor metadata without changing the data. Another example of the transparent node is the `ReplaceSliceNode` that changes the input tensor but returns the pointer to the input so does not have the output memory.

The buffer memory is the memory used for intermediate computation. It is required during the node evaluation but is not needed at another time.

Memory allocator is closely related to the evaluation control flow. Consider some two nodes $A$ and $B$ of the model. If $A$ and $B$ are never executed concurrently then the memory allocator can take advantage of this fact and reuse the buffer memory of $A$ for the buffer of $B$. Generally, the memory allocation should preserve two rules:
- Constant memory of the node should be available and unchanged during the model lifetime.
- Buffer memory of the node should be available during the execution of the node.
- Ouptut memory of the node should be available during the execution of the nodes that use the output of the node.

Note that these rules imply some memory-performance trade-offs. Greater buffer memory reuse leads to the smaller memory consumption but can reduce the parallelism of the execution since two nodes sharing the buffer or the output memory cannot be executed concurrently. In HamKaas we execute all the nodes sequentially since it is simpler, so we can take advantage of the buffer memory reuse.

Let's start implementing memory allocator. Open the file `allocator.h` in the `hamkaas/cpp` directory. You need to implement the `TAllocator` class. Note, that this class does not really allocate the memory but just computes the memory layout of the nodes.

The intended usage of the allocator is following: during the model compilation, the allocator is created and the model graph is traversed. When buffer or output memory is required, compiler calls `Allocate` method and gets the offset of the memory in future memory block. When memory is no longer needed (e.g., the node is executed), the compiler calls `Free` method to free the memory. After the graph traversal is completed, `GetWorkingSetSize` is called to get the amount of memory required for the memory execution. After that, the real memory block of size `GetWorkingSetSize` is allocated at address `base` and each node receives the memory address `base + offset` where `offset` is the value returned by the `Allocate` method.

Implement the `TAllocator` class. If you want to keep things simple, you can implement the allocator that just allocates the memory sequentially. If you want to have more challedge, you can implement some advanced memory allocation strategy that reuses memory after `Free` calls. In order to avoid thinking about the memory alignment, all the allocations should be aligned to 256 bytes (i.e. for all allocations `offset % 256 = 0`). `cudaMalloc` also aligns all allocations by 256 bytes, so after adding `base` addresses will be still aligned.

When you completed the allocator, you can run simple unittest by executing `make test-allocator` in `hamkaas/cpp` directory. If you see `All tests passed!`, good job! You can move to the next step.

After the allocator is completed, you can use it to allocate buffer and output memory for the nodes. Open the `model.cpp`, you will implement `TModel::AllocateMemory()` method.

`EvaluationOrder_` is the vector that contains all non-input nodes in the topological order. During the inference, the nodes are executed in this order. Remember, that HamKaas never executes two nodes concurrently. `InputNodes_` is the vector that contains all input nodes. The following methods of the `TNodeBase` are needed for the memory allocations:

- `GetConstantMemorySize()` returns the amount of constant memory required for the node.
- `GetBufferSize()` returns the amount of buffer memory required for the node.
- `GetOutputSize()` returns the amount of output memory required for the node.
- `SetConstantMemory(char* constantMemory)`, `SetBuffer(char* buffer)`, `SetOutput(char* output)` set the buffers for the node.
- `GetOutputOwner()` returns the node that allocated the memory for the output of this node. Typically, it is the node itself (so, `node->GetOutputOwner() == node`) but for the aforementioned transparent nodes it can be other node. For example if there is a `MatMulNode` and `ReshapeNode` that takes the result of the `MatMulNode` as an input, then the `ReshapeNode` should return the `MatMulNode` as the output owner. This method is used to understand when some output memory can be freed.

To allocate the real memory for the nodes, you need to call `Device_->DeviceMalloc` method. `Device` is the abstraction used in HamKaas to encapsulate some device-specific operations. Right now, there is only one device - `CpuDevice` defined in `device.h` that is used in case of the CPU execution. You will implement `CudaDevice` for the GPU execution later. Do not forget to clean up the memory in the destructor!

When you are ready, compile the `libhamkaas.so` library by running `make debug` in the `hamkaas/cpp` directory. Then you can test your implementation by running the tests in the `hamkaas/python/tests` directory by running `pytest -sv -k TestCompilerCpu`. This will run all the HamKaas tests with models executing on the CPU. If all tests are passed, good job! Your compiler is now working!

## 03: Going CUDA!

Alright, our compiler is now able to execute the models on the CPU. Now, let's make it work on the GPU!

The first thing you need to do is to implement the `TCudeDevice` class in the `device.cpp` file. You should be already familiar with functions used for memory operations. Note, that stream is passed in order to `cudaMemcpyAsync` use it. Pay attention to the `sync` parameter of copy functions. In case if `sync = true` you should synchronize the stream after the copy operation using `cudaStreamSynchronize`.

Rebuild `libhamkaas.so` library and run CUDA tests by running `pytest -sv -k TestCompilerCuda` in the `hamkaas/python/tests` directory. You should get `test_constant_node`, `test_input_node`, `test_invalid_input`, `test_buffer_tensor`, `test_slice_node` and `test_reshape` working. Other tests should not work yet and this is normal since you did not write GPU implementations of the nodes yet.

Now you will implement the missing nodes. I suggest implementing them one-by-one and testing them after each implementation.

Let's start with the `TPointwiseNode`, this is the base class for `SumNode`, `HadamardProductNode`, `ReLUNode` and `SiLUNode`. The common base is extracted because these nodes are pretty similar in terms of the implementation, they are all element-wise operations. The difference between these nodes is the arity: `SumNode` and `HadamardProductNode` are binary operations while `ReLUNode` and `SiLUNode` are unary operations.

Your goal is to implement `Pointwise` function in the `kernels.cu` file. You can find the usage of this function in the `TPointwiseNode::EvaluateGpu` function. Implementation of the `Pointwise` function should be similar to the kernels you've implemented in the previous labs. Pay attention to the broadcasting. You can find the reference CPU implementation in the `TPoinwiseNode::EvaluateCpu` function. For the performance reasons you can implement two kernels for pointwise operations: one for the case if broadcasting is needed and one for the case if it is not needed however this is completely optional. When you are ready, rebuild the library and run the CUDA tests. You should find `test_sum`, `test_relu`, `test_silu`, `test_hadamard_product` and `test_fibonacci` working.

The next step is `MatMulNode`. There are two options here. You can implement everything yourself in `kernels.h`/`kernels.cu` files or you can use cuBLAS library as we did before. The first one should be pretty straightforward while the second one is more challenging.

If you will use cuBLAS for this task, you will need `cublasGemmBatchedEx` function. Note, that cuBLAS uses column-major order, so you will need to use `CUBLAS_OP_T` for both matrices and the result will be transposed as well, so you will need to transpose it back. This can be done using `cublasSgeam` function (take a look at the documentation to find out how). To use cuBLAS you will also need to allocate some intermediate memory on GPU (for example, for the array of pointers to matrices). To do it, set the proper intermediate memory amount in `GetBufferSize` method of the `TMatMulNode` class and get the pointer to the memory in the `SetBuffer` method. Note, that addresses of the outputs of the input nodes are unknown during the `SetBuffer` call, so initialize the intermediate memory in the `Initialize` method. You can find example of the intermediate memory management in the `PointwiseNode` where the intermediate memory is used to store shape of inputs.

Also, note that all the evaluation of GPU model is done within the stream in order to record it into the CUDA graph, so you will need to pass the stream from the `TEvaluationContext` to the cuBLAS using `cublasSetStream` function.

When you are ready, rebuild the library and run the CUDA tests. You should find `test_matmul` working.

Our next destination is the `TPermuteNode` that changes the order of the dimensions of the tensor. As with the `MatMulNode` there are two options here. The first one is to implement the kernel yourself: each thread of the kernel should copy one value from the input tensor to the output tensor according to the permutation. The second (and more challenging) one is to use efficient implementation by NVIDIA that is availabe in the [cuTENSOR](https://docs.nvidia.com/cuda/cutensor/latest/index.html) library. To apply the permutation to tensor you can use [cutensorPermute](https://docs.nvidia.com/cuda/cutensor/latest/api/cutensor.html#cutensorpermute) function.

After you finish with the `TPermuteNode` you can run the tests again. You should find `test_permute` working.

The final one is the `TReplaceSliceNode`. One of the efficient ways to copy the data within the GPU is to use `cudaMemcpy` with the `cudaMemcpyDeviceToDevice` flag. However, this requires to have static addresses of the source and destination memory in order to fuse this operation into the CUDA graph. In our case this is false since the begin and end offsets are input parameters. So, in order to implement this node you will need to implement the kernel that copies the data. Write this kernel in `kernels.h/kernels.cu` and use it in `TReplaceSliceNode::EvaluateGpu` function. When you are ready, rebuild the library and run the CUDA tests. You should find all the tests finally working.

Congratulations! You have completed this lab and now your HamKaas complier is able to execute the models on the CPU and GPU!
