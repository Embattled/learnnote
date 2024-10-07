# 1. CUDA C Best Practices Guide

与实际问题相关的 issue 的解决白皮书


https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/contents.html

## 1.1. Preface


# 2. Heterogeneous Computing



# 3. Application Profiling

# 4. Parallelizing Your Application
<!-- 完 -->
完成了上述的基本练习后, 开发人员需要实际进行代码并行化.  

根据原始代码, 调用 GPU 优化后的库 例如 `cuBLAS, cuFFT, Thrust` 
或者添加预处理指令以作为并行编译器的提示

程序或许需要进行重构, 来使其暴露原本不可见的某种并行性, 以便提高程序的性能.  
CUDA 的编程语言 (CUDA C++, CUDA Fortran) 则是用于使这些表达尽可能的简单

# 5. Getting Started


# 6. Getting the Right Answer

# 7. Optimizing CUDA Applications

# 8. Performance Metrics

# 9. Memory Optimizations

内存优化是性能最重要的领域, 通过最大化内存带宽来最大化硬件的使用.  

主旨是: 尽可能多的使用快速内存, 减少慢速内存的使用

## 9.1. Data Transfer Between Host and Device


## 9.2. Device Memory Spaces

CUDA 可以访问的 device memory 也根据层级分为了多种

Register
Local
Shared
Global
Constant
Texture

### 9.2.1. Coalesced Access to Global Memory

CUDA 编程的一个重要内存访问因素 `coalescing of global memory accesses`

高优先级: warp 线程的全局内存加载和存储由设备合并为尽可能少的事务  

coalescing 的访问需求取决于设备的计算能力, 并记录在 CUDA C++ Programming Guide 中
* 计算能力高于等于 6.0 的新设备: 
  * 服务所有线程所需要的 32-byte 单位的transactions
  * 线程的并发访问的 transactions数量等同于 ↑
* 计算能力为 5.2 的设备, 可以选择利用 L1 缓存, 缓存的 transactions 数量等于 128-byte 对其段的数量
  * 高于等于 6.0 的设备其实也默认启用了 L1 缓存, 但无论负载是否在 L1 之中, 访问单位都是 32-byte

对于 GDDR 内存, 当 ECC 打开的时候, coalesced access 是更加重要的, 特别是对于将数据写回 global momory 的时候

#### 9.2.1.1. A Simple Access Pattern

# Execution Configuration Optimizations

# Instruction Optimization

# Control Flow

