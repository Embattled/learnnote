# 1. Open Computing Language

OpenCL 是
* 第一个面向异构系统通用目的并行编程 的 开放式, 免费标准
* 是一个统一的编程环境
* 是便于软件开发人员为  高性能计算服务器, 手持设备, 普通桌面计算系统等 编写高效轻便的代码
* 广泛用于 多核心处理器 图形处理器  Cell类型架构  数字信号处理器 (DSP) 等其他 `并行` 处理器   

终归的定义为 : 为异构平台编写程序的框架, 可由CPU, GPU或其他类型的处理器组成  
* 由一门基于C99的用于编写 kernels 的语言(又称 OpenCL C)和一组用于定义并控制平台的 API 组成
* 提供了基于任务分割和基于数据分割的并行计算机制

和另外两个开放的工业标准类似
* OpenGL 用于三位图形和计算机三维图形方面
* OpenAL 用于计算机音频方面
* OpenCL 拓展了GPU在计算方面的能力


https://www.khronos.org/opencl/


## 1.1. OpenCL 的历史  

2008 年由苹果公司于WWWDC大会上发布, 拥有商标权 , 随后提交至 Khronos Group 并成立了 GPU 通用计算开放行业标准工作组.  
* 2008年11月18日, 工作组完成了 1.0 版本
* 2010年6月14日, OpenCL 1.1 发布
* 2011年11月15日, 1.2 发布
* 2013年11月19日, 2.0 发布 
* 2017年, 2.2 发布

2020 年9月30日 OpenCL 3.0 发布
* 3.0 在某种程度上开了历史倒车
* 2.X 的各种新功能在 3.0 中变为了可选

苹果于 macOS 10.14 中放弃了 OpenCL , 推行自家的 Metal API  
NVIDIA 目前有更加成熟且社区活跃的 CUDA API  
AMD 目前只支持 OpenCL 1.2  


## 1.2. 异构平台并行计算  

面向异构平台的应用都必须有以下步骤
1. 发现构成异构系统的组件
2. 探查组件的特征, 是的软件能够适应不同硬件单元的特定特性
3. 建立 指令块(内核)
4. 建立并管理计算中涉及的内存对象
5. 在系统中正确的组件上按正确的顺序执行内核
6. 收集最终结果

OpenCL程序员必须显式地定义平台, 上下文, 以及在不同设备上调度工作.  具体通过 OpenCL 的一系列 API 再加上一个面向内核的编程环境来完成, 分而治之的策略来实现以上所有步骤  
* 平台模型 (platform model)  : 异构系统的高层描述
* 执行模型 (execution model) : 指令流在异构平台上执行的抽象表示
* 内存模型 (memory model)    : OpenCL 中内存区域以及 计算期间内存区域的交互方法
* 编程模型 (programming model) : 设计算法来实现应用的计算高层抽象   


一个典型的OpenCL宿主机程序, 需要
1. 定义上下文, 命令队列, 内存, 程序对象和宿主机所需要的数据结构
2. 内存对象从宿主机转移到设备上
3. 内核参数关联到内存对象
4. 然后提交到命令队列执行
5. 内核完成工作时, 计算中生成的内存对象可能会再复制到宿主机。

### 1.2.1. 平台模型  

定义了使用 OpenCL 的计算平台的一个高层表示, 包括一个常驻的 宿主机(host), 与一个或者多个 OpenCL 设备连接, 设备是具体的指令流计算的地方, OpenCL 通常称计算设备, 可以是 CPU, GPU, DSP 以及其他任何支持的处理器  

OpenCL 设备可以划分成计算单元, 而计算单元则有 一个或者多个处理单元构成

### 1.2.2. 执行模型

OpenCL 的模型由两个部分构成 , 在宿主机上执行的 宿主机程序, 在OpenCL 设备上执行的 内核(一个或者多个)  

内核程序: 通常是一些简单的函数, 将输入内存对象转换为输出内存对象
* OpenCL 有两种内核
* OpenCL 内核 : 由 OpenCL C 语言编写并由 OpenCL 编译器编译的函数, OpenCL 标准的所有实现都必须支持 OpenCL 内核
* 原生内核 : OpenCL 之外创建的函数, 在OpenCL中可以通过一个指针函数来访问


执行流程:
* 内核定义在 宿主机上 , 由宿主机发出命令, 提交内核到 OpenCL 设备上来具体执行
* OpenCL 运行时会创建一个整数索引空间
* 索引空间里的每一个索引坐标都对应执行内核的一个实例, 各个执行内核的实例成为一个工作项, 每个工作项的全局 ID 就是索引坐标
* 工作项(work_item)使用内核顶一个的相同的指令序列, 并根据全局 ID 以及分支来产生不同的计算结果
* 工作项组织成工作组, 提供了对索引空间更粗粒度的管理, 跨越整个全局索引空间, 工作项可以由自身的全局ID或者工作组 ID + 工作组内部局部ID 来确定具体的执行内容

上下文: 由宿主机定义的一个环境, 内核在环境中定义和执行
* OpenCL 应用的计算工作是在 OpenCL 设备上进行的, 但是辅助 计算工作(内核) 的其他内容 (上下文) 都是由宿主机来完成的, 具体包括  
* 设备 (device) : OpenCL 设备集合
* 内核 (kernel) : 在 OpenCL 设备上运行的 OpenCL 函数
* 程序对象 (program object) : 实现内核的程序源代码和可执行函数
* 内存对象 (memory object) : 内存中对 OpenCL 设备可见的对象, 包含由内核实例处理的值

命令队列 (command-queue):  
* 由宿主机创建一系列命令提交到命令队列中, 在上下文定义完成后被关联到一个 OpenCL 设备上, 调度这些命令在关联设备上执行  
* 内核执行命令 (kernel execution command) : 在 OpenCL 设备的处理单元上执行内核
* 内存命令 (memory command) : 在宿主机和不同内存对象之间传递数据, 在内存对象之间移动数据, 或者将内存对象映射到宿主机地址空间, 或者从宿主机地址空间解映射
* 同步命令 (synchronization command) : 对命令执行的顺序施加约束  


### 1.2.3. 内存模型

OpenCL 定义了两种类型的内存对象: 
* 缓冲区对象 (buffer object) : 内核可用的一个连续的内存区, 可以将数据结构映射到该缓冲区并同过指针访问
  * 内存连续线性排布
  * 针对多维数据需要对索引降维成1维才能访问
  * 用户使用地址偏移来访问
  * 支持较多类型数据
  * 可读写
  * 需要用户进行边界保护
* 图像对象 (image object) : 仅限于存储图像, 图像存储格式可以进行优化来满足一个特定 OpenCL 设备的需要
  * 针对二三维数据, 内存不一定连续
  * 使用 Sampler 来进行数据访问
  * 不支持自定义类型
  * 低版本不支持读写同时操作
  * 不需要手动进行边界处理

OpenCL 定义了 5 种不同的内存区域, 有不同的抽象机制
* 宿主机内存 (host memory) : 只对宿主机可见, 通常指 CPU端内存, OpenCL 只定义了宿主机内存与 OpenCL 对象和构造如何交互
* 全局内存 (global memory) : 允许读写多有工作组中的所有工作项, 通常指显存, 工作项可以读写全局内存中一个内存对象的任何元素,
* 常量内存 (constant memory) : 由宿主机分配并初始化放在常量内存中的内存对象, 这些对象对于工作项来说是只读的
* 局部内存 (local memory) : 对于工作组来说是局部的, 存放一个工作组中所有工作项共享的变量, 可以实现为 OpenCL 设备上的专用内存区域, 也可以映射到全局内存中的某个区段
* 私有内存 (private memory) : 是一个工作项私有的区域, 私有内存中定义的变量对其他工作项不可见  

### 1.2.4. 编程模型

任务并行和数据并行

数据并行编程模型: 适合元素可以并发更新的数据结构
* 工作组中工作项的数据并行再加上工作组层次的数据并行
* 显示模式 (explicit model) : 程序员显示地定义工作组的大小
* 隐式模型 (implicit model) : 程序员只需要定义 NDRange, 由系统自动选择工作组

任务并行编程模型 : 不是 OpenCL 的主要目标, 但是支持任务并行算法
* 将任务定义为单个工作项执行的内核
* 不考虑其他内核使用的 NRRange


## 1.3. OpenCL 标准构成

OpenCL 框架可以划分成以下组成部分
* OpenCL 平台API : 平台 API 定义了宿主机程序发现 OpenCL 设备所用的函数以及这些函数的功能, 以及为OpenCL应用创建上下文的函数
* OpenCL 运行时函数 : 管理上下文来创建队列以及运行时发生的其他操作, 例如, 将命令提交到命令队列
* OpenCL 编程语言 : 基于 ISOC99 标注的一个拓展子集, 通常称为 OpenCL C


# 2. OpenCL 编译环境

## 2.1. Getting started with OpenCL on Ubuntu Linux

OpenCL-SDK : To build native OpenCL applications, one will minimally need
* C or C++ compiler
* The OpenCL headers : The C and optionally the C++ headers
* An Installable Client Driver (ICD) Loader : Shared objects library (`libOpenCL.so`)



## 2.2. OpenCL Reference

https://www.khronos.org/files/opencl30-reference-guide.pdf

总的来说, OpenCL 3.0 的参考被分成了几个部分

各个版本的文档链接
https://registry.khronos.org/OpenCL/



# 3. OpenCL API Reference 2
# 4. OpenCL API Reference 3

API 的参考 

## 4.1. Kernel Objects

kernel object 是 `__kernel` 函数的封装体, 提供了获取用于具体的 OpenCL 运行的参数的接口

它定义了不同于设备所物理支持的最大参数, 从软件层面上来一定程度上限制程序  

### 4.1.1. Create kernel objects


### 4.1.2. Kernel arguments and queries

询问 kernel 的参数  


```cpp
cl_int clGetKernelWorkGroupInfo (
  cl_kernel kernel,
  cl_device_id device,
  cl_kernel_work_group_info param_name,
  size_t param_value_size, void *param_value,
  size_t *param_value_size_ret)


```

