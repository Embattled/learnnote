# Android Native Development Kit (NDK)

一个用于在 Android 项目中使用 C/C++ 代码的开发组件  
* provides platform libraries that allow you to manage native activities
* access physical device components, such as sensors and touch input. 


## ndk-build


## Address Sanitizer (asan)

一种基于 编译器 的工具, 用于检测内存相关的 bug
* 检测 stack 和 global objects 的overflows 
* 占用内存少
* 处理快
* !! 不能检测内存泄漏以及访问未定义内存

https://developer.android.com/ndk/guides/asan