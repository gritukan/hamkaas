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

## 01: Dynamic Linkage.

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