# HamKaas: Build a Simple Inference Compiler

Have you ever been wondered how modern compilers for deep learning work? Or do you just want to get some hands-on experience with CUDA programming? Then this may be for you. This repository consists of a 5 labs that starts from a simple CUDA programs and ends with a simple compiler capable of running LLaMA 2 7b. You definitely won't become an expert in CUDA computing after this, but hopefully you will get a basic understanding of the concepts behind modern deep learning.

# Disclaimer

I have just finished this project, so the code and especially texts were not tested enough and bugs are likely. If you do not want to take a course with potential bugs, I suggest to wait for a while until it becomes more stable.

# Prerequisites

You need to have a basic knowledge of C++ and Python programming. A basic understanding of deep learning is a plus but not required. No prior experience with CUDA is needed.

You will also need an access to the host with a CUDA-compatible GPU with NVIDIA CUDA Toolkit installed. We will check if everything is set up correctly in the beginning of the first lab.

# Labs

The course consists of series of labs. Technically you can do them in any order (except for the lab 4 and lab 5 because the latter depends on the former), but it is recommended to do them in order.

In case if you got stuck on something, feel free to discuss it in the [Discord channel](https://discord.gg/CuftjcJr). Please do not create GitHub in this case. Use them only for found issues and feature requests.

[Lab 1: CUDA Basics](lab1/README.md). In this lab you will learn the basics of CUDA programming and write you first kernels.

[Lab 2: GPU Performance](lab2/README.md). In this lab you will learn how to profile your CUDA programs. Also you will learn about the mordern GPU architecture and use it to optimize your programs.

[Lab 3: CUDA Libraries](lab3/README.md). In this lab you will learn about CuBLAS and CuDNN libraries and will implement a simple neural network inference using them.

[Lab 4: HamKaas Part 1](lab4/README.md). In this lab you will start working on the HamKaas compiler and will learn about the basic concepts behind compilers.

[Lab 5: HamKaas Part 2](lab5/README.md). In this lab you will implement an optimizer for your compiler. At the end, you will add new operations to your compiler and will be able to run LLaMA 2 7b.

Lab 6: Distributed Inference. This lab is not ready yet, consider liking this [issue](https://github.com/gritukan/hamkaas/issues/2) if you are interested in it. This lab will be about NCCL and distributed deep learning algorithms.

# FAQ

## Why HamKaas?

Ham is a ham in Dutch and Kaas is a cheese. There are ham kaas croissants sold near my work in the Netherlands. My colleague thinks that they are tasteless while I like them for their simplicity.

That reflects the simplicity of the compiler we are going to build. It is simple but its simplicity is good for the educational purposes.

## Can I complete this course without access to a CUDA-compatible GPU?

Unfortunately, no. You need an access to the host with a CUDA-compatible GPU in order to run the programs you write. You can get an access to such a host by using cloud providers.

The good news is that you do not need to have a powerful GPU, every GPU that supports CUDA will be enough for all the labs expect the last involving LLaMA 2 7b, however almost any GPU will be enough for that lab as well.

## It is possible to write CUDA code on Python. Why do we use C++?

I consider C++ to be a better language for such courses because it is more low-level and closer to the hardware.

However, I think that it is possible to do everything required in this course on Python, so if you want to have the same course but Python-first, consider liking this [issue](https://github.com/gritukan/hamkaas/issues/1). I am not sure if I will have time to do it, but understanding the demand is useful anyway :)

# Contributing and Discussions

This course is pretty young, so I am open to any suggestions and contributions.

If you have any questions (either about solving lab or a project in general), feel free to ask them in the [Discord channel](https://discord.gg/CuftjcJr).

If you have found a bug, please create an issue (or pull request!) in the repository.

If you have an idea of a new lab or task, great! Please create an issue with the idea and we will discuss it.

# License

[MIT](LICENSE)
