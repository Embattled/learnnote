# 1. Android Development

在一个标准安卓项目中:  
Types of modules include:
* Android app modules
* Library modules
* Google App Engine modules


# 2. Android Studio

是 安卓官方的 IDE
* 以 IntelliJ IDEA 为基础构建而成


# 3. Android SDK software

Android SDK 中包含了开发应用所需的多个软件包

* 可以使用 Android Studio 的 SDK 管理器或 sdkmanager 命令行工具来安装和更新每个软件包

## 3.1. Android SDK 构建工具 (build-tools)


## 3.2. Android SDK 平台工具 (plantform-tools)

Android SDK Platform Tools 是 Android SDK 的一个组件
* 与 Android 平台进行交互的工具
* 解锁设备的引导加载程序并为其刷入新的系统映像
* 某些新功能仅适用于较新的 Android 版本, 但这些工具是向后兼容的


### 3.2.1. adb Android debug bridge

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