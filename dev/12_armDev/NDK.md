# Android Native Development Kit (NDK)

一个用于在 Android 项目中使用 C/C++ 代码的开发组件  
* provides platform libraries that allow you to manage native activities
* access physical device components, such as sensors and touch input. 
* 用于实现平台之间可移植的应用
* 重复使用既存的 C/C++ 库
* 在特定情况下 对于计算密集型应用有更好的性能


NDK 开发工作的相关组件
* 原生共享库    : NDK 从 C/C++ 源代码构建的`.so` 库文件
* 原生静态库    : NDK 从 C/C++ 源代码构建的 `.a` 库文件, 可以将静态库迭代关联到其他库
* Java 原生接口(JNI)  : JNI 是 Java 和 C++ 组件互相通信的接口, 详细信息参照 Java 原生接口规范
* 应用二进制接口(ABI)  : ABI 可以非常精确地定义应用的机器代码在运行时如何与系统交互

# ndk-build

使用 NDK 的基于 Make 的构建系统构建一个项目, 只通过 `Android.mk` 和 `Application.mk` 两个文件配置整个项目  

## Android.mk

位于项目 `jni` 目录的子目录中 , 用于描述 源文件和共享库, 内容上是一个 `makefile` 的片段

Android.mk 用于
* 定义 `Application.mk`
* 构建系统和环境变量 以外的 项目层级 变量
* Android.mk 的语法支持将源文件分组为 "模块"

要掌握jni, 就必须熟练掌握Android.mk的语法规范
* 指定诸如编译生成so库名
* 引用的头文件目录
* 需要编译的`.c/.cpp`文件和`.a`静态库文件

[Android.mk Doc in Android developers](https://developer.android.com/ndk/guides/android_mk#npv)

Android.mk 的基本构成: 
* 以 LOCAL_PATH 为开始, 包含在两个 include 中间的 LOCAL 变量为一个模块
* Android.mk 中可以定义复数个这样的模块, 用于同时编译多个目标模块

```makefile
# 以下为一个模块的构成
LOCAL_PATH := $(call my-dir)  
include $(CLEAR_VARS)  
................  
LOCAL_xxx       := xxx  
LOCAL_MODULE    := hello-jni  
LOCAL_SRC_FILES := hello-jni.c  
LOCAL_xxx       := xxx  
................  
include $(BUILD_SHARED_LIBRARY)

```

* `LOCAL_PATH`            : 获取该 .mk 的路径, `$(call my-dir)` 是NDK内部的函数
* `include $(CLEAR_VARS)` : 用于清空当前环境变量里除了 `LOCAL_PATH` 以外所有的 `LOCAL_*` 变量
* `LOCAL_*` 变量:
  * `MODULE` : 必须变量, 用以表示要生成的一个模块, 名字唯一且不带空格
    * 编译系统会自动给生成的文件添加 `lib` 前缀 和`.so / .a` 后缀
    * 如果模块被定义成了 `libabc` 会被识别, 生成文件不会带两个 lib, 系统只给生成的库文件添加后缀
    * 可以再定义一个 `LOCAL_MODULE_FILENAME` 变量来强制更改生成的模块文件名称
  * `MODULE_PATH` : 生成的模块的目标地址
    * `TARGET_OUT` (默认)  : `out/target/product/generic/system`
    * `TARGET_ROOT_OUT`    : `out/target/product/generic/root`
    * NDK 有很多预定义的类似的宏, 用于将生成的模块输出到不同的目录
  * `SRC_FILES`   : 包含要编译的所有 `C/C++` 文件, 空格分开, 不需要 `.h` (自动导入依赖)
* `include $(BUILD_xxx_xxx)` : 最终执行 NDK 的默认脚本, 会收集mk文件内定义的 `LOCAL_*` 变量并生成最终的模块
  * NDK 定义了几个不同的宏, 用于指向不同的 `GNU Makefile Script` 决定最终的编译输出
  * `BUILD_STATIC_LIBRARY`    : 编译为静态库
  * `BUILD_SHARED_LIBRARY`    : 编译为动态库
  * `BUILD_EXECUTABLE`        : 编译为可执行的 Native C 程序
  * `PREBUILT_*`              : 该模块已经预先编译 (不需要编译)
    * 此时 `LOCAL_SRC_FILES` 里应该是链接库文件而不是代码源文件
    * `PREBUILT_STATIC_LIBRARY` 
    * `PREBUILT_STATIC_LIBRARY`


## Application.mk

Application.mk 指定 ndk-build 的项目级设置  
默认情况下, 它位于应用项目目录中的 jni/Application.mk 下

# Address Sanitizer (asan)

一种基于 编译器 的工具, 用于检测内存相关的 bug
* 检测 stack 和 global objects 的overflows 
* 占用内存少
* 处理快
* !! 不能检测内存泄漏以及访问未定义内存

https://developer.android.com/ndk/guides/asan