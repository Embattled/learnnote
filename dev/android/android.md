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

## JNI层的 java 源文件

编写带有 `native` 声明的方法的java类
* native 是java啥关键字不记得了 放置

要注意:
* 因为只是声明, 所以IDE报错也没关系

## javac 编译 生成 h头文件

## Android.mk

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





# 4. Android SDK software

Android SDK 中包含了开发应用所需的多个软件包

* 可以使用 Android Studio 的 SDK 管理器或 sdkmanager 命令行工具来安装和更新每个软件包

## 4.1. Android SDK 构建工具 (build-tools)


## 4.2. Android SDK 平台工具 (plantform-tools)

Android SDK Platform Tools 是 Android SDK 的一个组件
* 与 Android 平台进行交互的工具
* 解锁设备的引导加载程序并为其刷入新的系统映像
* 某些新功能仅适用于较新的 Android 版本, 但这些工具是向后兼容的


### 4.2.1. adb Android debug bridge

功能多样的命令行工具:
* 与设备进行通信
* 用于执行各种设备操作
* 提供对 Unix shell（可用来在设备上运行各种命令）的访问权
* 是一种客户端-服务器程序
  * 客户端: 在开发机器上运行, 通过发出 adb 命令从命令行终端调用客户端
  * 服务器: 在开发机器上作为后台进程运行,  用于管理客户端与守护程序之间的通信
  * 守护程序 (adbd) : 用于在设备上运行命令, 守护程序在每个设备上作为后台进程运行
  * `Android 11` 后: 允许直接通过 WIFI 来无线连接 adb 调试. 



0. 简单的工作原理
   1. 启动某个 adb 客户端时:
      1. 检查是否有 adb 服务器进程正在运行, 如果没有->启动服务器进程
      2. 服务器在启动后会与本地 TCP 端口 `5037` 绑定
      3. 并监听 adb 客户端发出的命令 , 所有 adb 客户端均通过端口 5037 与 adb 服务器通信
   2. 服务器会与所有正在运行的设备建立连接:
      1. 通过扫描 5555 到 5585 之间 (该范围供前 16 个模拟器使用) 的 `奇数` 号端口查找模拟器
      2. 服务器一旦发现 adb 守护程序(adbd), 便会与相应的端口建立连接
   3. 每个 adbd 都占用连续数字的 2 个端口 (一对)
      1. `奇数` 号端口 : 用于与 adb 连接    `5555  5557 ...`
      2. `偶数` 号端口 : 用于与控制台连接    `5554  5556 ...`


1. `devices` 用于验证已连接的设备:
   * `-l` 用于输出详细的信息 目录为下:
      * 序列号 : adb 为设备创建的标识
      * 设备状态: `offline` `device` `nodevice`
      * 设备说明: 包括设备的详细信息


adb 全局参数, 适用于所有命令, 在命令各自的说明里不再赘述:
* `-s 序列号`       : 指定要执行命令的设备, 在连接到多台设备的时候非常有用


2. `install [options] path` : 将 path 指定的 apk 安装
   * `-r`   : 保留软件数据的重新安装
   * `-t`   : 安装测试用的 apk (通过 ide的 build apk 生成的测试版apk 不能直接安装)
   * `f`    : 在内部系统内存上安装
   * ...

3. `shell`  : 用于给设备发送 shell 命令
   * Android 提供了大多数常见的 Unix 命令行工具, 用 `adb shell ls /system/bin` 查看有没有熟悉的想要使用的命令
   * `adb [-d |-e | -s serial_number] shell shell_command` 发送单条命令
   * `adb [-d |-e | -s serial_number] shell` 启动交互式 shell

4. `push / pull`    : 用于设备与电脑直接的文件转移
   * `adb pull remote local` : 从设备(手机) 中提取文件
   * `adb push local remote` : 从本地推送文件到设备(手机)
   * remote 的地址, 直接用根目录开始的地址即可, 不需要加什么端口或设备名