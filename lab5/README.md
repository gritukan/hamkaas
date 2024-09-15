In this lab we will add optimizer to the HamKaas compiler and will add some more nodes to make it possible to infer LLaMA2.

# 01: cuDNN optimizer

NOTE: By default, HamKaas is compiled without cuDNN support to make compilation faster. To enable cuDNN support use `make debug-cudnn/make release-cudnn` instead of `make debug/make release`. Also, you need to download cuDNN frontend to the `cpp` directory using `git clone https://github.com/NVIDIA/cudnn-frontend` just like you did in the lab3.

In the lab3, we learned about the cuDNN library and how to use it to create fused kernels for GEMM and pointwise operations.

Remember, that cuDNN is capable of some graphs into the kernel but not all, so it is up to the HamKaas compiler to split the graph into the parts that can be subsequently fused by cuDNN.

Open the `cudnn_optimizer.cpp` file and look at the `RunCudnnOptimizer`. This function takes the graph as an input, and returns the graph with some nodes replaced by the `TCudnnNode`.

Function `GetSubgraphsToFuse` is already implemented for you. It runs some heuristics graph traversals to find the subgraphs such that cuDNN can fuse.

After subgraphs are found nodes are replaced in graph using `ReplaceNodes` function. This part is also implemented for you.

The part that is left for you is to implement the `Compile` method of the `TCudnnNode` that builds a cuDNN graph and implement `EvaluateGpu` method that runs it.

Here are some hints:
* `Nodes_` may have only `TMatMulNode`, `TSumNode`, `THadamardProductNode`, `TReLUNode` and `TSiLUNode` nodes. All of these are supported by cuDNN.
* Be careful about the inputs of the `TCudnnNode` since they are not set up yet. Note, that in the `Compile` method you still do not know the addresses of the outputs of the input nodes. These adresses will become known in the `Initialize` method.
* cuDNN graphs require memory for the intermediate results that is called workspace. You can allocate memory for the workspace using `GetBufferSize/SetBuffer`.
* Remember everything you learned about cuDNN in the lab3. All tensors should have dimension 3. In the solution of the lab3 you already have 80% of the code required to finish this task.
* Good luck!

When you are done, rebuild libhamkaas.so using `make debug-cudnn` and run cuDNN tests using `pytest -sv -k TestCompilerCudnn`. Note, that these tests are slow because of the slow cuDNN compilation. If all tests pass, congratulations!

The next step is implementing compilation cache. In many models, the same subgraphs appear many times. Compiling the same subgraph many times is a waste of time, so let's add a cache that will reuse compiled subgraphs.

The cache is the `std::unordered_map<std::string, TCudnnCompilationResult>&` argument that is passed to the `Compile` method. The key here is the trace of the graph (i.e. some text representation of the graph that describes its structure). The value is the result of the compilation. `Graph` is the compiled `cuDNN` graph, `InputTensors` is a map from input name to the input node descriptor and the `OutputTensor` is the output node descriptor.

To create a graph trace, you will need to traverse the graph and concatenate the descriptions of the nodes. Make sure to enumerate the nodes in order to store the shape in the trace.

When caching is implemented, run tests again to test if everything works correctly.

# 02: LLaMA

This is your final challenge. To infer the LLaMA you will need to add some new nodes to the compiler. In the frontend you will need to implement methods `complex_hadamard_product`, `rms_norm` and `sliced_softmax`. Here are the descriptions of the nodes:

#### ComplexHadamardProductNode
* Syntax: `ComplexHadamardProductNode(lhs, rhs)`
* Description: interprets a tensor with shape `[..., 2]` as a tensor of complex numbers of shape `[...]` with `[...][0]` being the real part of a complex number and `[...][1]` being the imaginary part. The node computes the pointwise product of these two complex tensors (i.e. `(a + bi) * (c + di) = (ac - bd) + (ad + bc)i`). Note that the broadcasting is supported with the similar behavior as in the `HadamardProductNode`.
* Inputs: `lhs` and `rhs` are tensors of type `float32` and last dimension being 2. All other axis must be compatible for broadcasting just like `HadamardProductNode`.
* Output: a tensor of type `float32` with the same shape as the `lhs` tensor.

#### RmsNormNode
* Syntax: `RmsNormNode(input, weights)`
* Description: computes the root mean square norm of the input tensor with given weights. [Here](https://github.com/tairov/llama2.py/blob/4bf4ac89c0ff0d154a7e3602c1279039ecb51dac/llama2.py#L93-L105) is the reference implementation in Python.
* Inputs: `input` and `weights` are tensors of type `float32` and 1D shape `[n]`.
* Output: a tensor of type `float32` with shape `[n]`.

#### SlicedSoftmaxNode
* Syntax: `SlicedSoftmaxNode(input, prefix_size)`
* Description:
    - If `input` is a vector of shape `[n]`, computes the softmax of the first `prefix_size` elements of the vector and returns the vector of size `n` with the softmax of the first `prefix_size` elements and the rest of the elements unchanged. [Here](https://github.com/tairov/llama2.py/blob/4bf4ac89c0ff0d154a7e3602c1279039ecb51dac/llama2.py#L108-L122) is the reference implementation of the softmax (but not sliced softmax!) in Python.
    - If `input` is a matrix of shape `[m, n]`, interprets this matrix as `m` vectors of size `n` and computes the sliced softmax for each of the vectors independently.
* Inputs: `input` is a tensor of type `float32` and shape `[n]` or `[m, n]`. `prefix_size` is a constant `int64` value.
* Output: a tensor of type `float32` with the same shape as the `input` tensor.

You will need to add the support of these nodes in multiple places. In the frontend you will need to create new node classes and serialize these nodes to script in `traverse_node`. In the backend you will need to parse these nodes in `parser.cpp` and implement new nodes in `node.h/node.cpp`. Note, that `ComplexHadamardProductNode` is a pointwise operation, so it may be derived from the `TPointwiseNode`.

When you are done, remove the fuse returns in `test_complex_hadamard_product`, `test_rms_norm` and `test_sliced_softmax` in `test_compiler.py` and run all the tests using `pytest -sv .`

If all tests pass, congratulations! Your compiler should be now able to infer LLaMA2. To do it, you will need to run `llama2.py` from the `python` directory.

This script is the fork of the [llama2.py](https://github.com/tairov/llama2.py) by Aydyn Tairov which is based on [llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy.

Note, that cuDNN is disabled by default because complitation is quite slow and inference performance improvement is not that big.

To run it, you may use instructions from the [llama2.c](https://github.com/karpathy/llama2.c). You can download tokenizer using `wget https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin` and, for example, tinystories 15M model using `https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin` and run the script using `python3 llama2.py stories110M.bin 0.8 256 "Donald was a little duck"`. If you implemented everything correctly, you should see some adequate output.

The script is fully compatible with `llama2.py` and `llama2.c`, so you can use the same models including original llama2 models.

That's it. You have completed your journey from the simple CUDA kernel to running LLM on the compiler. Good job!
