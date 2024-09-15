Hi, and welcome to the course!

In this lab you will learn how to program CUDA, a parallel computing platform and programming model developed by NVIDIA for GPUs. It is used to execute code on the GPU and is used in different machine learning frameworks like PyTorch and JAX. HamKaas also uses CUDA for computations, so it is important to 

We will go through the basic concepts of CUDA and will write some programs to get you started.

# 00: Check setup

Before we start, let's check if you have set up all the nessessary tools for this lab correctly. You are exepected to have an access to a machine with a CUDA-compatible GPU with drivers and CUDA toolkit installed.

To check if everything is set up correctly, run the following command in the terminal:

```bash
make 00-test
```

This command will compile and run a simple CUDA program located in `00.cu` file. It simply adds increases a floating-point number by 1.0 on GPU using CUDA. If everything is set up correctly, you should see the following output:

```
nvcc --std=c++17 -o 00 00.cu
./00
Test passed (x = -42)
Test passed (x = 0)
Test passed (x = 3.14)
Test passed (x = 1e+100)
All tests passed
```

If everything is set up correctly, congratulations! You are ready to start the lab. If you see an error, try to figure out the problem yourself. During the course, you will face many problems requiring additional reading and research, so you are expected to be able to deal with problems like this.

## 01: Hello, CUDA!

In this task, you will write your first CUDA kernel that sums two vectors of floating-points numbers of the same length. In other words, you are given vectors $a$ and $b$ of length $n$ and are required to compute the vector $c$ of length $n$ such that $c_i = a_i + b_i$ for all $i$.

The code for this task is located in `01.cu` file. Let's have a look at what is inside.

The function `AddVectorsGpu` is responsible for GPU vector addition and making it work is our goal for now. `AddVectorsCpu`, `DoTest` and `main` are used for testing and you don't need to modify them. Let's look at the `AddVectorsGpu` function closer.

At first, it allocates three arrays on the GPU using `cudaMalloc` function. Note, that GPU is a separate device with its own memory, so in order to process data on GPU you have to allocate memory on it. Also, note that all the CUDA calls are wrapped into a `CUDA_CHECK_ERROR` macro. This is because every CUDA call can fail and you should always check if it did. This macro checks for error and prints an error message and fails program if there is one. You can find its implementation in `common.h` header but it is not important for now. You can read more about CUDA error handling in the [official documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html).

Next, we need to copy data from the host memory to the GPU memory. This is done using `cudaMemcpy` function. It has syntax similar to regular `memcpy` function, but it requires additional arguments to specify the direction of the copy (from host to device or vice versa).

Finally, we are ready to call CUDA kernel. Kernel is simply a function that is executed on the GPU. Since GPU is massively parallel, the kernel is executed on many threads in parallel. `AddVectorsKernel<<<1, a.size()>>>` syntax means that we are going to start `a.size()` threads in a single block (we will talk about blocks later). It worth mentioning that the kernel is executed asynchronously, so the control is returned to the host code immediately after the kernel is launched, it does not wait for all threads to finish. `CUDA_CHECK_KERNEL` macro is used to report an error if the kernel launch failed.

After the kernel is executed, we copy the result back to the host memory and free the GPU memory. Note, that memcpy performs synchronization, so there is no race between kernel execution and memory copy. For explicit synchronization, you can use `cudaDeviceSynchronize` function.

Now it's your time to implement `AddVectorsKernel`. As being said, this is a function that is executed on the GPU many times in different threads. We want every thread to compute the sum of two elements of the input vectors. To distinguish between threads, you can use `threadIdx.x` variable which is a built-in variable running from `0` to `a.size() - 1` in our case.

When you are done, run the following command to test your implementation:

```bash
make 01-test
```

If you see `All tests passed`, great job!

<details>
<summary> Solution spoiler! </summary>

```cpp
__global__ void AddVectorsKernel(double* a, double* b, double* c)
{
    int index = threadIdx.x;
    c[index] = a[index] + b[index];
}
```

</details>

## 02: Going 2D!

In this task, you will write a kernel that computes the sum of two matrices of the same size. The code for this task is located in `02.cu` file.

The code is quite similar to the vector addition task, but now we have to deal with matrices (i.e. 2D arrays). In CUDA, matrices are usually stored in a [row-major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order), that is a matrix with $N$ rows and $M$ columns is stored as a 1D array of size $n \times m$ where the element $(i, j)$ is stored at the index $i \times m + j$.

For the simplicity of work with 2D and 3D arrays, CUDA supports multidimensional arrays of threads. Kernel launch code looks like this for our problem:

```cpp
dim3 threadsPerBlock(n, m);
AddMatricesKernel<<<1, threadsPerBlock>>>(gpuA, gpuB, gpuC, m);
```

It makes CUDA to launch `AddMatricesKernel` in $n \times m$ threads in a single block. Threads are organized in a 2D grid in this case. You can access the thread index in each dimension using `threadIdx.x` and `threadIdx.y` variables. Note, that CUDA supports thread arrays of up to 3 dimensions.

Now, implement `AddMatricesKernel` kernel and test in by running `make 02-test` command. If you see `All tests passed`, great job!

<details>
<summary> Solution spoiler! </summary>

```cpp
__global__ void AddVectorsKernel(double* a, double* b, double* c)
{
    int index = threadIdx.x;
    c[index] = a[index] + b[index];
}
```

</details>

## 03: Thread Blocks

In this task, you will write a kernel that sums two vectors of the same length again. However, this time you will use multiple blocks of threads to do the job. The code for this task is located in `03.cu` file.

In CUDA, threads are organized in blocks. Threads in the same block can communicate with each other using shared memory. Threads in different blocks are cannot communicate with each other. The number of threads in a block is limited by the hardware and is usually 1024 or less.

In our case, threads are not communicating with each other, so it may seem that we can just create $n$ blocks with a single thread in each block. However, this is not optimal because it does not utilize GPU fully. We will talk about it in the next lab. For now, we will use 8 threads per block.

Take a look at how the kernel is launched in this case:

```cpp
constexpr int ThreadsPerBlock = 8;
int blocksPerGrid = (a.size() + ThreadsPerBlock - 1) / ThreadsPerBlock;
AddVectors<<<blocksPerGrid, ThreadsPerBlock>>>(a.size(), gpuA, gpuB, gpuC);
```

The first argument tells how many blocks of threads should be launched and the second tells how many threads should be in each block. Note, that both of the arguments can be of `dim3` type to arrange block or threads in arrays.

Now, it's your turn to implement `AddVectors` kernel. You may find useful the built-in variable `blockDim.x` which contains the number of threads in the block. Note, that in case if the number of elements in the input vectors is not divisible by the number of threads in the block, the total number of threads launched will be greater than the number of elements in the vectors. You should carefully handle such case to avoid out-of-bounds access.

When you are done, test your soultion with `make 03-test`. If you see `All tests passed`, great job!

<details>
<summary> Solution spoiler! </summary>

```cpp
__global__ void AddVectors(int n, double* a, double* b, double* c)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}
```

</details>

## 04: Shared Memory

In this task you will write a kernel that swaps adjecent elements of a vector using shared memory. The code for this task is located in `04.cu` file.

Shared memory is a special kind of memory that is shared between threads in the same block. It is much faster than global memory, but it is limited in size (usually 16KB-192KB depending on the particular hardware). Shared memory is used to exchange data between threads in the same block and to reduce the number of global memory accesses.

For this problem you will implement a kernel that is launched in $n \over 2$ blocks with $2$ threads each. For each block, a shared memory buffer of size $2$ is allocated. Kernel for element $i$ firstly store the element $a_i$ into shared memory, then waits for all threads in the block to finish, and after that stores value of the element $a_{i + 1}$ taken from the shared memory to the $a_i$ of the output vector. That is, every thread makes one read from the global memory and one write to the global memory.

To allocate a fixed amount of the shared memory `__shared__ double buffer[2];` syntax is used.

For the threads synchronization `__syncthreads()` function is used. It makes all threads in the block to wait until all of them reach the synchronization point.

When you are done, test your solution with `make 04-test`. If you see `All tests passed`, great job!

<details>
<summary> Solution spoiler! </summary>

```cpp
__global__ void SwapAdjacentKernel(double* data)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int localIndex = threadIdx.x;

    __shared__ double buffer[2];
    buffer[localIndex] = data[index];

    __syncthreads();

    data[index] = buffer[1 - localIndex];
}
```

</details>

## 05: Your First Solo Flight!

In this task, you will implement a [SiLU](https://en.wikipedia.org/wiki/Swish_function) activation function kernel. The code for this task is located in `05.cu` file.

For this task you will need an exponential function `exp` which is available in CUDA math library. You can learn more about supported functions in the [official documentation](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html).

Unlike previous tasks, this time `SiLUGpu` is not implemented for you, so you need to implement both kernel and host code. Try not just to copy the code from the previous tasks, but to understand what you are doing.

When you are done, test your solution with `make 05-test`. If you see `All tests passed`, great job!

<details>
<summary> Solution spoiler! </summary>

```cpp
__global__ void SiluKernel(double* a, int n)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n) {
        double x = a[index];
        a[index] = x / (1 + exp(-x));
    }
}

std::vector<double> SiLUGpu(std::vector<double> data)
{
    double* gpuA;
    CUDA_CHECK_ERROR(cudaMalloc(&gpuA, data.size() * sizeof(double)));
    CUDA_CHECK_ERROR(cudaMemcpy(gpuA, data.data(), data.size() * sizeof(double), cudaMemcpyHostToDevice));

    constexpr int MaxThreadsPerBlock = 256;
    int blocksPerGrid = (data.size() + MaxThreadsPerBlock - 1) / MaxThreadsPerBlock;
    SiluKernel<<<blocksPerGrid, MaxThreadsPerBlock>>>(gpuA, data.size());
    CUDA_CHECK_KERNEL();

    std::vector<double> result(data.size());
    CUDA_CHECK_ERROR(cudaMemcpy(result.data(), gpuA, data.size() * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaFree(gpuA));

    return result;
}
```

</details>

## 06: Max Pooling

In this task, you will implement a max pooling operation kernel. The code for this task is located in `06.cu` file.

Max pooling is a common operation in convolutional neural networks. You can learn about it [here](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html). In this problem you will implement a kernel of size $4 \times 4$ and stride 1. Formally, for an input matrix $n \times m$ you need to compute an output matrix of the same size satisfying $\text{out}_{i, j} = \max_{i}^{min(n - 1, i + 3)} \max_{j}^{min(m - 1, j + 3)} \text{in}_{i, j}$. Refer to $MaxPoolingCpu$ for a simple implementation.

Hints:
* You may assume that all matrix elements are non-negative, so you can use zero as a negative infinity.
* Use 256 threads per block with each block processing $16 \times 16$ submatrix.
* Use shared memory to store the submatrix and then calculate maximums from values in the shared memory.
* To find $16 \times 16$ submatrix of the output matrix you need a larger (which size?) submatrix of the input matrix. Try to write a code in such a way that each thread does at most $3$ global memory accesses. If you want to add a little bit challenge, try to do it with $2$ accesses.

When you are done, test your solution with `make 06-test`. If you see `All tests passed`, great job!

<details>
<summary> Solution spoiler! </summary>

```cpp
template <int KernelSize, int BlockDimensionSize>
__global__ void MaxPoolingKernel(double* input, double* output, int n, int m)
{
    int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;
    if (globalX >= n || globalY >= m) {
        return;
    }

    int localX = threadIdx.x;
    int localY = threadIdx.y;

    __shared__ double buffer[BlockDimensionSize + KernelSize][BlockDimensionSize + KernelSize];

    buffer[localX][localY] = input[globalX * m + globalY];

    bool needExtraX = (globalX + KernelSize < n && localX + KernelSize >= BlockDimensionSize);
    if (needExtraX) {
        buffer[localX + KernelSize][localY] = input[(globalX + KernelSize) * m + globalY];
    }

    bool needExtraY = (globalY + KernelSize < m && localY + KernelSize >= BlockDimensionSize);
    if (needExtraY) {
        buffer[localX][localY + KernelSize] = input[globalX * m + (globalY + KernelSize)];
    }

    if (needExtraX && needExtraY) {
        buffer[localX + KernelSize][localY + KernelSize] = input[(globalX + KernelSize) * m + (globalY + KernelSize)];
    }

    __syncthreads();

    double result = 0.0;
    for (int dx = 0; dx < KernelSize; dx++) {
        for (int dy = 0; dy < KernelSize; dy++) {
            if (globalX + dx < n && globalY + dy < m) {
                result = max(result, buffer[localX + dx][localY + dy]);
            }
        }
    }

    output[globalX * m + globalY] = result;
}
```

</details>
