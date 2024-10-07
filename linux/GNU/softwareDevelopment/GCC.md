- [1. GCC, the GNU Compiler Collection](#1-gcc-the-gnu-compiler-collection)
- [2. 编译](#2-编译)
  - [2.1. -std 标准选择](#21--std-标准选择)
  - [2.2. 链接](#22-链接)
    - [2.2.1. 标准库的链接](#221-标准库的链接)
    - [2.2.2. 自己的链接库](#222-自己的链接库)
    - [2.2.3. 动态链接库的显式调用](#223-动态链接库的显式调用)
    - [2.2.4. 解决找不到库的问题](#224-解决找不到库的问题)
- [3. GCC Command Options](#3-gcc-command-options)
  - [3.1. Option Summary](#31-option-summary)
  - [3.2. Options Controlling the Kind of Output](#32-options-controlling-the-kind-of-output)
  - [3.3. Compiling C++ Programs](#33-compiling-c-programs)
  - [3.4. Options Controlling C Dialect](#34-options-controlling-c-dialect)
  - [3.5. Options Controlling C++ Dialect](#35-options-controlling-c-dialect)
  - [3.6. Options to Control Diagnostic Messages Formatting](#36-options-to-control-diagnostic-messages-formatting)
  - [3.7. Options to Request or Suppress Warnings](#37-options-to-request-or-suppress-warnings)
  - [3.8. Options That Control Optimization](#38-options-that-control-optimization)
  - [3.9. Machine-Dependent Options - 平台相关的参数](#39-machine-dependent-options---平台相关的参数)
    - [3.9.1. ARM Options - 手机端常用的 ARM 架构的选项](#391-arm-options---手机端常用的-arm-架构的选项)
      - [3.9.1.1. ARM march=](#3911-arm-march)
    - [3.9.2. GNU/Linux Options - GNU Linux 系统命令](#392-gnulinux-options---gnu-linux-系统命令)
    - [3.9.3. x86 Options - x86 平台选项](#393-x86-options---x86-平台选项)
  - [3.10. Options for Linking - 链接选项](#310-options-for-linking---链接选项)
  - [3.11. Options for Code Generation Conventions](#311-options-for-code-generation-conventions)

# 1. GCC, the GNU Compiler Collection

https://gcc.gnu.org/


GCC 全名: GNU C Complier, 即在 GNU 计划中诞生的 C 语言编译器.
* 早期的 GCC 确实只用于编译 C 代码, 但是随着版本更替, 现在的 GCC 已经能支持 C++, GO, Objective-C 等多语言
* 目前最新的 GCC 全名的定义被更改为 `GNU Complier Collection`
* G++ 是 GCC 的一个版本, 是自动链接C++库的一个编译器版本

GCC可以自动识别文件扩展名, 匹配类型. GCC 是一个通用的多语言的编译器 
* 如果当前文件的扩展名和识别表不符，只需要借助 `-x` 选项(小写)指明当前文件的类型即可。
* 为 gcc 指令添加 `-xc` 选项，表明当前 demo 为 C 语言程序文件  
* `-x` 指令还可以后跟 c-header(C语言头文件), c++(C++源文件), c++-header(C++程序头文件)等选项

正因为GCC的多语言支持性, 因此 G++ 更像是一个只针对 C++ 语言的编译器
1. 后缀为 `.c` 的文件, gcc当成 `C` 代码 , 而g++会当成 `CPP` 代码
   gcc不会定义 `__cplusplus` 宏 ,而g++会 , 这个宏标志着编译器会把代码按照C还是C++语法来解释,如果后缀为`.c`,并且采用gcc编译器,啧该宏就是未定义的
2. gcc不能**自动**和CPP的库链接, 所以一种解决方法是用g++来完成链接


对于C++代码的编译  
```shell
$ g++ HelloWorld.cpp -o a.out # 使用g++可以直接编译链接
$ gcc -xc++ -lstdc++ -shared-libgcc HelloWorld.cpp -o a.out # 使用gcc需要加上链接的库
```

可以这样认为，`g++` 指令就等同于`gcc -xc++ -lstdc++ -shared-libgcc`指令  
* `-xc++` :指定源代码为 C++ 语言
* `-lstdc++` : 链接C++ std 库
* `-shared-libgcc` : 不懂
* 所以, 使用G++更多的是为了省去链接C++库的命令行参数


# 2. 编译

| gcc/g++指令选项 | 功 能                                                                               |
| --------------- | ----------------------------------------------------------------------------------- |
| -E(大写)        | 预处理指定的源文件，不进行编译。                                                    |
| -S(大写)        | 编译指定的源文件，但是不进行汇编。                                                  |
| -c              | 编译, 汇编指定的源文件，但是不进行链接。                                            |
| -o              | 指定生成文件的文件名。                                                              |
| -ansi           | 对于 C 语言程序来说，其等价于 -std=c90；对于 C++ 程序来说，其等价于 -std=c++98。    |
| -std=           | 手动指令编程语言所遵循的标准，例如 c89, c90, c++98, c++11 等。                      |
| -l`library`     | library 表示要搜索的库文件的名称, 建议 -l 和库文件名之间不使用空格，比如 -lstdc++。 |

通常情况下, 可以选择单命令编译和分步编译, 单命令用于快速编译单个简短程序, 分步编译一般用于 makefile 的书写中, 用于大型程序节省编译时间




## 2.1. -std 标准选择

不同版本的 GCC 编译器，**默认使用**的标准版本也不尽相同。  

对于编译 C, C++ 程序来说，借助 `-std` 选项即可手动控制 GCC 编译程序时所使用的编译标准  
`gcc/g++ -std=编译标准`

注意 1z 是 17 的别称

**c**  
| stand                          | descript                                |
| ------------------------------ | --------------------------------------- |
| -std=c90 or -std=iso9899:1990  | 称为C89或C90                            |
| -std=iso9899:199409            | 称为C94或C95                            |
| -std=c99 or -std=iso9899:1999. | 1999年发布的 ISO/IEC 9899:1999，称为C99 |
| -std=c11 or -std=iso9899:2011  | 称为C11                                 |
| -std=gnu90                     | C90和GNU扩展                            |
| -std=gnu99                     | C99和GNU扩展                            |
| -std=gnu11                     | C11和GNU扩展                            |


**c++**  
| stand                     | descript       |
| ------------------------- | -------------- |
| -std=c++98, or -std=c++03 | 称为 C++98     |
| -std=c++11                | 称为C++11      |
| -std=c++14                | 称为C++14      |
| -std=c++17 or -std=c++1z  | 称为C++17      |
| -std=gnu++98              | C++98和GNU扩展 |
| -std=gnu++11              | C++11和GNU扩展 |
| -std=gnu++14              | C++14和GNU扩展 |
| -std=gnu++1z              | C++17和GNU扩展 |


## 2.2. 链接

将多个库文件链接起来
* 对于静态库 : Linux 中用 `.a` 表示, Windows 中为 `.lib`
* 对于动态库 : Linux 中为 `.so`, Windows 为 `.dll`

### 2.2.1. 标准库的链接

标准库的大部分函数通常放在文件 `libc.a`  中(文件名后缀.a代表“achieve”，译为“获取”)   
或者放在用于共享的动态链接文件 `libc.so` 中(文件名后缀.so代表“share object”，译为“共享对象”)  
这些链接库一般位于 /lib/ 或 /usr/lib/，或者位于 GCC 默认搜索的其他目录。


当使用 GCC 编译和链接程序时，GCC 默认会链接 libc.a 或者 libc.so，但是对于其他的库(例如`非标准库, 第三方库`等)，就需要手动添加。  


标准头文件` <math.h>` 对应的数学库默认也不会被链接  
数学库的文件名是 libm.a。前缀lib和后缀.a是标准的，m是基本名称  
GCC 会在-l选项后紧跟着的基本名称的基础上自动添加这些前缀, 后缀，本例中，基本名称为 m  
`gcc main.c -o main.out -lm`  

通常，GCC 会自动在标准库目录中搜索文件，例如 /usr/lib，如果想链接其它目录中的库，就得特别指明。有三种方式可以链接在 GCC 搜索路径以外的链接库  
1. 把链接库作为一般的目标文件，为 GCC 指定该链接库的完整路径与文件名
   如果链接库名为 `libm.a`，并且位于 `/usr/lib` 目录，那么下面的命令会让 GCC 编译 main.c，然后将 libm.a 链接到 main.o  
   `gcc main.c -o main.out /usr/lib/libm.a`  
2. 使用-L选项，为 GCC 增加另一个搜索链接库的目录, 再在其后面接上要链接的库名  
   `gcc main.c -o main.out -L/usr/lib -lm`  
   可以使用多个-L选项，或者在一个-L选项内使用冒号分割的路径列表  
3. 把包括所需链接库的目录加到环境变量 LIBRARYPATH 中


```sh
# 对于cpp文件 , 填入库相关的路径
gcc -lstdc++ hello.cpp

# 对于c文件 
gcc hello.c -lm -L /usr/lib -I /usr/include
```
* -lm 指的是libm.so或libm.a这个函数库文件；
* -L 后面接的路径是刚才上面那个函数库的搜索目录；
* -I 后面的是源码内的include文件所在的目录；



### 2.2.2. 自己的链接库

当把程序链接到一个`链接库`时，只会链接程序所用到的函数的目标文件。在已编译的目标文件之外，如果创建自己的链接库，可以使用 ar 命令。

1. 创建自己的静态库  
   可以被加工成静态库的文件:  
   * 源文件中只提供可以重复使用的代码，例如函数, 设计好的类等，不能包含 main 主函数
   * 源文件在实现具备模块功能的同时，还要提供访问它的接口，也就是`包含各个功能模块声明部分的头文件`  

    步骤
    1. 将所有指定的源文件，都编译成相应的目标文件
       `gcc -c sub.c add.c div.c`  
    2. 然后使用 ar 压缩指令，将生成的目标文件打包成静态链接库，其基本格式如下  
       `ar rcs 静态链接库名称 目标文件1 目标文件2 `  
       `ar rcs libmymath.a add.o sub.o div.o`  
       静态链接库的不能随意起名，需遵循如下的命名规则 `libxxx.a`  Windows 系统下，静态链接库的后缀名为 `.lib`  

   此时想使用自己的静态库 , 可以使用命令  
   ` gcc -static main.o libmymath.a`  
   使用动态库的命令
   `gcc main.c  libmymath.so -o main.exe`  
   `-static` 选项强制 GCC 编译器使用静态链接库  

   对于别的目录下的库, 使用组合命令 `-L` 和 `-l`  ,分别指定目录和库名  
  `gcc main.o -static -L /root/demo/ -lmymath`  
  
2. 创建自己的动态库  

    **直接使用源文件创建动态链接库**  
    `gcc -fpic -shared 源文件名... -o 动态链接库名`  
    * -shared 选项用于生成动态链接库
    * -fpic (还可写成 -fPIC): 令 GCC 编译器生成动态链接库(多个目标文件的压缩包)时，表示各目标文件中函数, 类等功能模块的地址使用相对地址，而非绝对地址。这样，无论将来链接库被加载到内存的什么位置，都可以正常使用
  

    **从目标文件编译成动态链接库**
    为了后续生成动态链接库并能正常使用，将源文件编译为目标文件时，也需要使用 -fpic 选项  
    `gcc -c -fpic add.c sub.c div.c`  

    ` gcc -shared add.o sub.o div.o -o libmymath.so`  

    **使用动态库进行链接的命令**  
    `gcc main.c  libmymath.so -o main.exe`  

   
   动态库应用程序的查看  
   `ldd main.exe`  可以查看查看当前文件在执行时需要用到的所有动态链接库，以及各个库文件的存储位置  
   
   运行由动态链接库生成的可执行文件时，必须确保程序在运行时可以找到这个动态链接库。常用的解决方案有如下几种：

   * 将链接库文件移动到标准库目录下(例如 /usr/lib, /usr/lib64, /lib, /lib64)；
   * 在终端输入 `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:xxx` 其中 xxx 为动态链接库文件的绝对存储路径
     (此方式仅当前终端有效，关闭终端后无效)；
   * 修改 ~/.bashrc 或~/.bash_profile 文件 在文件最后一行添加上一个方法的内容 即
     `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:xx` 
     保存之后，执行source .bashrc指令(此方式仅对当前登陆用户有效)。

### 2.2.3. 动态链接库的显式调用

动态链接库的调用方式有 2 种，分别是：
* 隐式调用(静态调用)：将动态链接库和其它源程序文件(或者目标文件)一起参与链接
  * 即上文的链接方法
* 显式调用(动态调用)：手动调用动态链接库中包含的资源，同时用完后要手动将资源释放
  * 显式调用动态链接库，更常应用于一些大型项目中
  * 显式调用动态链接库的过程，类似于使用 malloc() 和 free()(C++ 中使用 new 和 delete)管理动态内存空间
  * 需要时就申请，不需要时就将占用的资源释放。由此可见，显式调用动态链接库对内存的使用更加合理。

和隐式调用动态链接库不同，在 C/C++ 程序中显示调用动态链接库时，无需引入和动态链接库相关的头文件  
但与此同时，程序中需要引入另一个头文件，即` <dlfcn.h>` 头文件，因为要显式调用动态链接库，需要使用该头文件提供的一些函数。

`#include <dlfcn.h>`   相关笔记写在 cbasic.md 文档

这里需要添加 -ldl 选项(使用了 dlfcn.h后 , 该可执行程序需要 libdl.so 动态库的支持)
`gcc main.c -ldl -o main.exe`  

### 2.2.4. 解决找不到库的问题

1. 链接时 
   假设当前 mian.c 文件需要借助 libmymath.a 才能完成链接，则完成链接操作的 gcc 指令有以下 2 种写法
   * `gcc -static main.c libmymath.a -o main.exe`  
     GCC 编译器只会在当前目录中(这里为 demo 目录)查找 libmymath.a
   * `gcc -static main.c -lmymath -o main.exe`  
     GCC 编译器会按照如下顺序，依次到指定目录中查找所需库文件
     1. 如果 gcc 指令使用 -L 选项指定了查找路径，则 GCC 编译器会优先选择去该路径下查找所需要的库文件
     2. 再到 Linux 系统中 LIBRARY_PATH 环境变量指定的路径中搜索需要的库文件
     3. 最后到 GCC 编译器默认的搜索路径(比如 /lib, /lib64, /usr/lib, /usr/lib64, /usr/local/lib, /usr/local/lib64 等，不同系统环境略有差异)中查找
   根据使用的方法不同移动相关的链接库位置解决问题

2. 运行时
   执行已生成的可执行文件时，如果 GCC 编译器提示找不到所需的库文件，这意味着 GCC 编译器无法找到支持可执行文件运行的某些动态库文件  
   GCC 编译器提供有 ldd 指令，借助该指令，我们可以明确知道某个可执行文件需要哪些动态库文件做支撑, 这些动态库文件是否已经找到, 各个动态库文件的具体存储路径等信息

   当 GCC 编译器运行可执行文件时，会按照如下的路径顺序搜索所需的动态库文件
   1. 如果在生成可执行文件时，用户使用了-Wl,-rpath=dir(其中 dir 表示要查找的具体路径，如果查找路径有多个，中间用 : 冒号分隔)选项指定动态库的搜索路径，则运行该文件时 GCC 会首先到指定的路径中查找所需的库文件；
   2. GCC 编译器会前往 LD_LIBRARY_PATH 环境变量指明的路径中查找所需的动态库文件
   3. GCC 编译器会前往 /ect/ld.so.conf 文件中指定的搜索路径查找动态库文件
   4. GCC 编译器会前往默认的搜索路径中(例如 /lib, /lib64, /usr/lib, /usr/lib64 等)中查找所需的动态库文件。
   
   注意!! 可执行文件的当前存储路径，并不在默认的搜索路径范围内，因此即便将动态库文件和可执行文件放在同一目录下，GCC 编译器也可能提示“找不到动态库”

   
   * 将动态库文件的存储路径，添加到 LD_LIBRARY_PATH 环境变量中。假设动态库文件存储在 /usr 目录中，通知执行export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr指令，即可实现此目的(此方式仅在当前命令行窗口中有效)；
   * 修改动态库文件的存储路径，即将其移动至 GCC 编译器默认的搜索路径中。
   * 修改~/.bashrc 或 ~/.bash_profile 文件，即在文件最后一行添加export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:xxx(xxx 为动态库文件的绝对存储路径)。保存之后，执行 source .bashrc 指令(此方式仅对当前登陆用户有效)。


# 3. GCC Command Options

<!-- 头部完 -->
大部分的 GCC 命令选项都会明确的表明适用的语言, 如果没有标注, 说明适用于所有GCC 支持语言.

GCC 的命令众多, 因此不要把多个命令参数打包, 例如不要把 `-d -v` 打包成 `-dv`, 会是完全不同的意思  

以 `-f` 和 `-W` 开头的命令有超级多, 并且大部分都有正负两种模式, 在文档中列出的都是 非默认模式 的那一边
* 例如 `-ffoo` 的 negative form 就是 `-fno-foo`
* `-W` 的 negative form 是 `-Wno-`

还有一些参数支持数字输入, 因此对于数字的有一些要求
* 必须是正数 unsigned
* 16进制的数字需要以 `0x` 开头
* 对于指定数据大小的参数, 可以可选的添加后缀 `kB KiB MB MiB GB GiB` etc

## 3.1. Option Summary

整合所有的命令并分类, 用于快速查找


Overall Options: 全局命令, 参照 3.2 Options Controlling the Kind of Output

C Language Options: C 语言命令 See Section 3.4 Options Controlling C Dialect]

C++ Language Options: See Section 3.5 Options Controlling C++ Dialect

## 3.2. Options Controlling the Kind of Output

GCC 的工作主要包括 预处理, 编译, 汇编,链接. `overall options` 可以指定这整个流程, 使得 gcc 只进行一部分的工作. 

`--version` : Display the version number and copyrights of the invoked GCC


从源代码生成可执行文件可以分为四个步骤，分别是预处理(Preprocessing), 编译(Compilation), 汇编(Assembly)和链接(Linking)  
* 较后的步骤指令会自动执行前面的步骤

1. 预处理  `-E`
   * `g++ -E demo.cpp -o demo.i`  
   * 必须使用 -o 选项将该结果输出到指定的 demo.i 文件  
   * Linux 系统中，通常用 ".i" 或者 ".ii" 作为 C++ 程序预处理后所得文件的后缀名  
2. 编译 `-S`   编译阶段针对的将不再是 demo.cpp 源文件，而是 demo.i 预处理文件  
   * `g++ -S demo.i`  
   *  生成对应的`.s`文件, Linux 发行版通常以 ".s" 作为其后缀名  
   *  和预处理阶段不同，即便这里不使用 -o 选项，编译结果也会输出到和预处理文件同名(后缀名改为 .s)的新建文件中  
3. 汇编  `-c`  
   * 通过给 g++ 指令添加 -c 选项，即可令 GCC 编译器仅对指定的汇编代码文件做汇编操  
   * ` g++ -c demo.s`  
   * 默认情况下汇编操作会自动生成一个和汇编代码文件名称相同, 后缀名为 `.o` 的二进制文件(又称为目标文件)
4. 链接  
   * 完成链接操作，并不需要给 g++ 添加任何选项  
   * 可以指定 `-o` 来指定输出的二进制可执行文件的名称, 一般 linux 下可执行文件的后缀都是 `.out`

对于 GCC 的自动文件类型识别来说, 每个编译步骤的文件后缀都不同, 即使是手动, 也尽量遵循该识别规则.  

下表仅仅列出了与 C/CPP 语言相关的 GCC 类型识别规则, 对于其他语言还有别的规则
| 拓展名                   | GCC的识别                |
| ------------------------ | ------------------------ |
| c                        | C源代码                  |
| cpp cp cc cxx CPP c++ C  | C++源代码                |
| i                        | 预处理, 未编译的C代码    |
| ii                       | 预处理后的 C++代码       |
| s                        | 编译后生成的汇编代码     |
| h                        | c,c++ 的头文件           |
| hpp h++ HPP hh H hxx tcc | C++ 头文件               |
| o                        | 编译后的目标文件(未链接) |


```shell
# 只编译 不链接
gcc -c hello.cpp 

# 链接或者直接编译出二进制
gcc hello.cpp

# 指定输出文件
gcc -o hello hello.cpp

# 严格编译 , 输出信息多
gcc hello.c -Wall
```

## 3.3. Compiling C++ Programs

When you compile C++ programs, you should invoke GCC as `g++` instead.



## 3.4. Options Controlling C Dialect

该章节描述源自于 C 的编译器命令, 同时支持从 C 语言派生的 C++, Objective-C, Objective-C++

`-ansi`

`-std=`


## 3.5. Options Controlling C++ Dialect

## 3.6. Options to Control Diagnostic Messages Formatting

用于去控制诊断信息的格式. 传统上, 诊断信息的格式与显示输出设备的方面无关.   
可以通过 `-f` 命令来控制诊断信息的格式信息, 例如每行多少个字符.  多久报告一次源代码位置信息, 某些语言可能不支持一些选项.  

## 3.7. Options to Request or Suppress Warnings

Warnings 属于诊断信息, 指明出来的警告在构造的本质上不是错误的, 但是存在风险, 或者可能存在错误. 

以下独立于语言的 warnging 命令选项不会启动特定的选项, 而是会控制 GCC 生成的诊断类型.

| 命令             | 功能                                                           |
| ---------------- | -------------------------------------------------------------- |
| `-fsyntax-only`  | 检查代码的语法错误, 除此之外不做任何事                         |
| `fmax-errors=n`  | 错误消息的最大数量为 n,超过n则编译器退出, 默认为 0 即不做限制. |
| `-Wfatal-errors` | 编译器出现第一个错误立马停止编译, 优先级高于 `fmax-errors=n`   |
| `-w`             | 禁止所有警告信息                                               |
| `-Werror`        | 让所有的警告变成 errors                                        |
| `-Werror=`       | 让指定的 warning 变成 error, 同时会隐式的启动对应的警告        |



## 3.8. Options That Control Optimization

超级长的一章 (70页), 命令多到不可能读完   

用于控制 GCC 在生成代码时候采取的优化措施  


默认行为, 无任何参数的时候, 编译器会将代码以最低的代价进行编译, 同时确保 debug 可以完整的运行并得到想要的结果
* Statements 是独立的, 可以在任何语句之间插入断点, 并更改对应变量的值   
* 可以正常的改变程序计数器, 使程序跳转到其他函数并得到期望的结果  

通过更改 optimization flags, 可以实现不同的 编译器优化策略
* 提高执行性能
* 编译的时间消耗
* 是否能够进行 debug

在编译的时候如果是从多个 source file 编译为 a single output, 则编译器可以理论上实现最多的算法信息利用  

GCC 的优化策略很多, 并不是所有的优化策略都能够通过 flag 控制, 只有拥有 flag 的被列在了该章节  

几乎所有的 flag 在单独指定的时候必须同时输入 `-O1` 以及以上 
* `-O0` 或者没有输入 `-O` 相关指定的时候, 这些 flag 都会被抑制
* 同理, `-Og` 也会抑制很大一部分的 flag
* 根据编译 target和 GCC 配置 的不同, `-O` 的效果也会有细微的不同, 可以通过 `-Q --help=optimizers` 来获取当前的 flag 效果


`-O` 系列的简要介绍 : 如果在命令中输入多个 -O 命令, 则只有最后的会生效 
| 命令       | 效果                                                                                                                        |
| ---------- | --------------------------------------------------------------------------------------------------------------------------- |
| `-O0`      | 降低编译时间, 同时确保 debug 可以运行, 默认选项                                                                             |
| `-O` `-O1` | 优化性编译 降低 code size 和执行时间, 编译消耗更多时间, 对于大函数的编译会消耗更多内存, 但是不会执行极度消耗编译时间的优化. |
| `-O2`      | 优化更多, 调用除了会产生 space-speed tradeoff 以外的, 几乎所有支持的优化策略                                                |
| `-O3`      | 基于 O2 的基础上, 进一步优化性能                                                                                            |
| `-Os`      | 基于 O2 的基础上, 取消了会增加 code size 的优化                                                                             |
| `-Oz`      | 基于 O2 的基础上, 更加积极的优化 code size 而不是 speed                                                                     |
| `-Ofast`   | 基于 O3 的基础上, 解除标准合规性, 会应用一些不是所有标准都支持的优化策略                                                    |
| `-Og`      | 基于 O1 的基础上去除所有会影响 debug 可行性的优化. 甚至优于 某些编译器上的 `-O0`, 因为有些编译器 O0 也不会保存 debug 信息   |

## 3.9. Machine-Dependent Options - 平台相关的参数

所有支持 GCC 的机器 (Architecture, operating system)  都可以拥有其独有的 option.  

该部分的 GCC 编译命令固定以 `-m` 开头  

以下是几十种不同平台的命令小章节, 只学习重要的

通用命令:
* `-march=[]`  似乎是指定架构的通用命令, 在多种平台存在

### 3.9.1. ARM Options - 手机端常用的 ARM 架构的选项


#### 3.9.1.1. ARM march=

`-march=name[+extension...]`   : 指定了 target ARM architecture, 确定编译后可以生成的 指令种类, 可以与 `-mcpu=` 结合或者代替使用

在通过 `name` 指定了 架构后, 还可以通过 `+extension` 的形式为该架构启用多种扩展, 拓展之间可能会有依赖. 存在 `+no**` 类型的拓展, 会以高优先级主动禁用某些功能, 而依赖于这些被禁用功能的扩展则会自动被一起禁用

```
Permissible names are: ‘armv4t’, ‘armv5t’, ‘armv5te’, ‘armv6’, ‘armv6j’,
‘armv6k’, ‘armv6kz’, ‘armv6t2’, ‘armv6z’, ‘armv6zk’, ‘armv7’, ‘armv7-a’,
‘armv7ve’, ‘armv8-a’, ‘armv8.1-a’, ‘armv8.2-a’, ‘armv8.3-a’, ‘armv8.4-a’,
‘armv8.5-a’, ‘armv8.6-a’, ‘armv9-a’, ‘armv7-r’, ‘armv8-r’, ‘armv6-m’,
‘armv6s-m’, ‘armv7-m’, ‘armv7e-m’, ‘armv8-m.base’, ‘armv8-m.main’,
‘armv8.1-m.main’, ‘armv9-a’, ‘iwmmxt’ and ‘iwmmxt2’.
```

以下是支持的架构对应的支持的拓展, 在上述中存在但在下列条框中不出现的架构则不支持任何拓展, 书写中省略 `+` 号  

* `armv8.2-a armv8.3-a`
  * `simd` : ARMv8.1-A Advanced SIMD and floating-point instructions.
  * `fp16` : 半精度浮点支持, 会同时启用 `Advanced SIMD` 和 `floating-point instructions`
  * `crypto` : 密码学指令, 会同时启用 `Advanced SIMD` 和 `floating-point instructions`
  * `dotprod` : 启用点乘 `Dot Product` 指令, 会同时启用 `Advanced SIMD`


### 3.9.2. GNU/Linux Options - GNU Linux 系统命令

### 3.9.3. x86 Options - x86 平台选项 


## 3.10. Options for Linking - 链接选项


## 3.11. Options for Code Generation Conventions

Code Generation COnventions 与机器无关, 文档给出的都是非默认的那一方
编写动态链接库时用的标志 `-fPIC`



* `-fpic`  : 生成 position-independent code (PIC) 用于 shared library.
  * 会把代码中的 constant addresses 打包成一个 global offset table (GOT).
  * 加载动态链接库的 dynamic loader 在执行程序的时候会解析该 GOT.
  * dynamic loader 不是 GCC 的一部分, 而是操作系统. 针对不同操作系统, GOT 表的上限都不同. 如果超过限制了, 使用 `-fPIC` 选项.
    * SPARC 8k
    * AArch64 28k
    * m68k, RS/6000 32k
    * x86  无限制
  * PIC 代码需要机器支持. 
    * x86 机器上, GCC 支持 System V, 不支持 Sun 386i
    * IBM RS/6000 机器的代码总是 PIC 的
  * 当该 flag 生效时 宏 `__pic__` 和 `__PIC__` 被设置成 1
* `-fPIC` : 生成 PIC 代码, 备用模式
  * 只在  AArch64, m68k, PowerPC and SPARC 机器上与 `fpic` 不同
  * 当该 flag 生效时 宏 `__pic__` 和 `__PIC__` 被设置成 2
  * 为了稳健, 应该总是使用 `fPIC` (Chatgpt 提供的示例)


