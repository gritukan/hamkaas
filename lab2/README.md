In this lab, you will start learning how to profile GPU workloads and optimize them.

## GPU Architecture Background

Before we start, let's first understand the GPU archirecture since it is important to work on CUDA workloads performance.

The GPU architecture is different from the classical computer architecture since they are designed to handle different workloads. CPUs are optimized to handle mostly serial workloads while GPUs are designed to execute the same code on the big amount of data. We will see how it affects the architecture.

The most basic component of the GPU is CUDA core which is similar to the CPU core in sense that it executes instructions in a serial manner. In terms of CUDA programming model CUDA core just executes a single thread of a kernel. The difference is that GPU has much more cores than CPU (for example, NVIDIA H100 almost 17k CUDA cores) but these cores are simpler than CPU cores since they are designed for a specific workload. Also, CUDA cores usually have smaller frequency than CPU cores, so on a single-threaded programs, CPU are usually faster than GPUs even on floating-point operations.

A Streaming Multiprocessor or just SM is a cluster of CUDA cores that can execute multiple threads concurrently. Also, SM contains a set of registers and shared memory that can be used by threads running on the cores of the SM. That is, block of the threads is always executed on a single SM. When a block is assigned to some SM, its threads are divided into warps. A warp is a group of 32 threads that are executed in a [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) manner. There are more than 32 cores in a SM, so SM can execute multiple warps simultaneously. Warp scheduler is responsible for selecting warps to execute on SM.

When warp is executed it is expected that all threads are executing the same instruction simultaneously. If some threads are executed different instructions, then the parallelism and thus the performance is decreased. This is called warp divergence. We will see examples of the warp divergence in this lab later.

Summarizing, GPU consists of a CUDA cores that are groupped into SMs. A block of threads is executed on a single SM and its threads are divided into warps. When warp is exectuted, its threads are running on SM cores.

Another important topic is GPU memory hierarchy. There are many types of memory on the GPU but now we will focus only on some of them.

Global memory is the main memory of the GPU. It is accessible from all the threads. In a previous lab, you already used global memory when allocated in via `cudaMalloc`. Global memory is the largest but slowest memory on the GPU, so it is important to minimize the number of accesses to it. In a modern NVIDIA H100 GPU, the global memory size is 80GB and the bandwidth is up to 3TB/s. This is much more than the CPU memory bandwidth but is still not sufficient for some workloads.

Shared memory is the memory that is located inside the SM and is shared between threads in the block. It is much faster than global memory but it is much smaller than global memory. In a modern NVIDIA H100 GPU, the shared memory size is 164KB per SM and the bandwidth can go up to tens of TB/s. It worth mentioning that shared memory is not homogeneous and is divided into multiple banks. During the single cycle only one operation per bank can be performed, so different threads should work with different banks to avoid bank conflicts resulting in performance degradation. We will see examples of bank conflicts in this lab later.

The fastest memory on the GPU are registers. Just like in the CPU, registers belong to a single core and thus accessible from the single thread. In a modern NVIDIA H100 GPU, the number of registers per CUDA core is 256.

Also similarly to CPU, there are caches on the GPU. L1 cache is located in the SM and is shared between all the cores (and threads) of it. It sits between SM and main memory and caches data from the global memory that is frequently accessed or was recently written. In NVIDIA H100 L1 cache and shared memory share the same space of the SM, so 256KB of the SM memory can be configurably shared between them.

L2 cache is located near the global memory and is shared between SMs. It caches data from the global memory that is frequently accessed by the SMs. In NVIDIA H100 L2 cache size is 40MB.

## NVIDIA Nsight

NVIDIA Nsight is a family of profiling tools by NVIDIA that are used to investigate the performance of the GPU workloads. In this lab, we will look at NVIDIA Nsight Systems and NVIDIA Nsight Compute.

NVIDIA Nsight Systems is a system-wide performance analyzer. It collects events from both CPU, GPU and GPU interconnect and puts them on a single timeline. It is useful to get a bird's eye view of the application performance and understand how wall time is distributed between different parts of the program and devices. On the other side, it does not provide detailed information about what happens inside the kernel like `perf` tool for the CPU workloads. To get a detailed view of the kernel execution, we will use another tool called NVIDIA Nsight Compute.

Both of these tools have excellent documentation (see [Nsight Systems](https://docs.nvidia.com/nsight-systems/) and [Nsight Compute](https://docs.nvidia.com/nsight-compute/)). It is definitely not required to read all of it to use these tools, but it is a good idea at least to look at the list of possible options and metrics that can be collected in order to understand the limitations of these tools. Simple usages of these tools will be shown in this lab later.

## 00: First steps with Nsight Systems

Now it is the time to make your hands dirty and profile something. In this task you will profile program that is located in file `00.cu`.

Take a look at the code. What does it do?

<details>
<summary> Answer </summary>

This code does the following procedure for 10 times. Allocate a vector of 1000 numbers, do some CPU processing, then do GPU processing. For the GPU processing data is copied to GPU and vice versa.

</details>

Now compile the code with the following command and run the profiling.

```bash
make 00
nsys profile -o 00.prof ./00
```

This will generate a file `00.prof.nsys-rep` with the profiling results. Download it locally if required and open it with NVIDIA Nsight Systems GUI.

You will see a timeline with the CPU and GPU events. Play with it to become comfortable with the tool. Try zooming timeline in and out, clicking on the events to get detailed information. Unroll sections for the CPU and GPU and look at them.

Look at the CUDA API section of the main thread. You will see alternating long `cudaMemcpy` calls and empty spaces with some other short calls.

What are the empty spaces?

<details>
<summary> Answer </summary>

This is a CPU processing. 
</details>

Why `cudaMemcpy` calls are so long? Processing should be longer, right?

<details>
<summary> Answer </summary>

`cudaMemcpy` performs synchronization between CPU and GPU that waits until all the previous GPU operations are finished. So it includes both the real data transfer and waiting for the kernel completion.

You can unroll CUDA HW section and see that most of the `cudaMemcpy` call time is spent on the GPU running kernel and only a small amount of time is spent for copying data.
</details>

Remove the body of the kernel and run profiling again. Find in the profiler that GPU time is mostly spent on host-to-device and device-to-host memory transfers.

Now, let's profile the kernel. Restore the kernel body and run the following command.
