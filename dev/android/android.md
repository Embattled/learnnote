# 1. Android Development

在一个标准安卓项目中:  
Types of modules include:
* Android app modules
* Library modules
* Google App Engine modules


安卓项目目录结构, 根据 IDE 不同有些微的差异

* `app/`          : 程序文件
  * `build/`      : 编译生成的结果目录, 类似于 bin
  * `libs/`       : 依赖库目录 (jar包)
  * `src/`        : 项目源文件
    * `androidTest/` : Android Test 用例, 对项目进行自动化测试
    * `test/`        : Unit Test 用例, 对项目进行自动化测试
    * `main/`     : 代码
      * `java/`   : java代码目录
      * `jni/`    : 原生代码目录 (C/C++)
      * `res/`    : 其他资源目录, 图像资源 布局资源, 菜单资源 
      * `libs/`   : 如果项目内容是库的话, 编译生成的 .so 放在这里   
    * `build.gradle` : app模块的 gradle 构建脚本

# 2. Android Studio

是 安卓官方的 IDE
* 以 IntelliJ IDEA 为基础构建而成

# 3. jni Java Native Interface 

Java1.1开始, JNI标准成为java平台的一部分 

通过使用 Java本地接口书写程序, 可以确保代码在不同的平台上方便移植
* 允许JAVA程序调用C/C++写的程序
* 书写步骤:
  1. 编写带有native声明的方法的java类
  2. 使用javac命令编译所编写的java类
  3. 然后使用javah(已被舍弃) + java类名生成扩展名为h的头文件
  4. 使用C/C++实现本地方法
  5. 将C/C++编写的文件生成动态连接库

## 3.1. JNI层的 java 源文件

编写带有 `native` 声明的方法的java类
* native 是java啥关键字不记得了 放置

要注意:
* 因为只是声明, 所以IDE报错也没关系

## 3.2. javac 编译 生成 h头文件

## 3.3. Android.mk

要掌握jni, 就必须熟练掌握Android.mk的语法规范
* Android.mk是Android提供的一种makefile文件
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

