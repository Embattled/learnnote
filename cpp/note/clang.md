# 1. What is clang?

* A language front-end and tooling infrastructure for languages in the C language family (C, C++, Objective C/C++, OpenCL, CUDA, and RenderScript) for the LLVM project. 
* Both a GCC-compatible compiler driver (clang) and an MSVC-compatible compiler driver (clang-cl.exe) are provided.

## 1.1. Introduction to Clang

1. Clang 是一个 C、C++、Objective-C 和 Objective-C++ 编程语言的编译器前端
2. 采用底层虚拟机（LLVM）作为后端
3. Clang 采用的是 `BSD` 协议的许可证，而 GCC 采用的是 `GPL` 协议，显然前者更为宽松
4. Clang 是一个高度模块化开发的轻量级编译器，编译速度快、占用内存小、有着友好的出错提示

## 1.2. Introduction to LLVM

**Low Level Virtual Machine**  
1. 但这个项目的范围并不局限于创建一个虚拟机
2. LLVM 成长之后已成为众多编译工具及低级工具技术的统称, 适用于 LLVM 下的所有项目，包括 LLVM 中介码、LLVM 除错工具、LLVM C++ 标准库等


LLVM 计划启动于 2000 年: 
1. `BSD` 许可来开发的开源的编译器框架系统
2. 基于 C++ 编写而成
3. 利用虚拟技术来优化以任意程序语言编写的程序的编译时间、链接时间、运行时间以及空闲时间
4. 最早以 C/C++ 为实现对象，对开发者保持开放，并兼容已有脚本。

* 苹果公司是 LLVM 计划的主要资助者
* LLVM 因其宽松的许可协议，更好的模块化、更清晰的架构
* 被苹果 IOS 开发工具、Facebook、Google 等各大公司采用，像 Swift、Rust 等语言都选择了以 LLVM 为后端。


LLVM和Clang的关系
* 广义的 LLVM 指的是一个完整的 LLVM 编译器框架系统，包括了前端、优化器、后端、众多的库函数以及很多的模块. 
    * 整体的编译器架构就是 LLVM 架构；Clang 大致可以对应到编译器的前端，主要处理一些和具体机器无关的针对语言的分析操作
* 狭义的 LLVM 则是聚焦于编译器后端功能的一系列模块和库，包括代码优化、代码生成、JIT 等。
    * 编译器的优化器和后端部分就是 LLVM 后端，即狭义的 LLVM



## 1.3. Why Clang?

Android NDK 在 Changelog 中提到：

    Everyone should be switching to Clang.
    GCC in the NDK is now deprecated.

1.  Android P 的逐步应用，越来越多的客户要求编译库时用 libc++ 来代替 libstdc++
    * libc++ 和 libstdc++ 都是 C++ 标准库
    * libc++ 是针对 Clang 编译器特别重写的 C++ 标准库
    * libstdc++ 则是 GCC 的对应 C++ 标准库
2. Android NDK 已在具体应用中放弃了 GCC，全面转向 Clang
    * Android NDK 从 r11 开始建议大家切换到 Clang，并且把 GCC 标记为 deprecated，将 GCC 版本锁定在 GCC 4.9 不再更新
    * 从 r13 起，默认使用 Clang 进行编译, Google 会一直等到 libc++ 足够稳定后再删掉 GCC
    * 在 r17 中宣称不再支持 GCC 并在后续的 r18 中删掉 GCC

