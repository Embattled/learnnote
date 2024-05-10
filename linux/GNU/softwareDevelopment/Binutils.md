- [1. GNU Binutils](#1-gnu-binutils)
- [2. GNU binary utilities (binutils)](#2-gnu-binary-utilities-binutils)
  - [2.1. ar  (archive)](#21-ar--archive)
    - [2.1.1. operation keyletter](#211-operation-keyletter)
  - [2.2. strings - 打印文件字符](#22-strings---打印文件字符)
- [3. ld - the GNU linker.](#3-ld---the-gnu-linker)
  - [3.1. Invocation - 命令行工具使用说明](#31-invocation---命令行工具使用说明)
    - [3.1.1. 通用链接命令](#311-通用链接命令)
      - [3.1.1.1. 路径搜索配置](#3111-路径搜索配置)
    - [3.1.2. Options Specific to i386 PE Targets - 主要的 PC 环境](#312-options-specific-to-i386-pe-targets---主要的-pc-环境)
    - [3.1.3. Environment Variables - ld 参照的环境变量](#313-environment-variables---ld-参照的环境变量)
  - [3.2. BFD 库](#32-bfd-库)
- [4. GNU gprof - Displays profiling information.](#4-gnu-gprof---displays-profiling-information)
- [5. GNU Gprofng - Next Generation Profiler](#5-gnu-gprofng---next-generation-profiler)

# 1. GNU Binutils 

超级大的 collection of binary tools.

很多被认为是 linux 基础的一些 `软件开发命令` 其实都属于该 GNU 的工具组

列表省略

其中的很多软件都使用了 `BFD, the Binary File Descriptor library`

整个 Binutils 被分成了多个部分, 就连文档也是分开的, 但是所有组件的版本都是统一的, 即 `GNU Binutils` 的版本
* Assembler (gas) 
* BFD Library (libbfd) 
* Binary Tools (binutils) 
* CTF Library (libctf)
* Linker (ld) 
* Next Generation Profiler (gprofng)
* Profiler (gprof) 
* Simple Frame Format Library (libsframe) 

# 2. GNU binary utilities (binutils)

更加细碎的各种指令, 作为一个个小的程序, 在开发中使用的非常广泛

## 2.1. ar  (archive)

The gnu `ar` program creates, modifies, and extracts from `archives`.
可以将多个 `.o` 目标文件打包成一个 `.a` 静态库文件

* An archive is a single file holding a collection of other files in a structure
* `archives` makes it possible to retrieve the original individual files (called members of the archive). 
* The original files’ contents, mode (permissions), timestamp, owner, and group are preserved in the archive, and can be restored on extraction. 
* `ar` 本身可以管理任意长度名称的 members, 但是为了与其他 GNU tools 的兼容性, 某些系统会对 ar 进行配置, 使得 members 的名称长度会受到限制, 通常是 15 或 16 个字符长度


```bash
ar [-]p[mod] [--plugin name] [--target bfdname] [--output dirname] [--record-libdeps libdeps] [relpos] [count] archive [member…]

ar -M [ <mri-script ]
```

执行 ar 最起码需要 2 个参数:
* 指定工作模式的 keyletter
* archive name
  * archive 后接的位置参数即为 member


参数:
* `--record-libdeps libdeps`

### 2.1.1. operation keyletter

文档中的 `[-]p[mod]` 代表 ar 最关键的工作模式, `-` 是可选的也可以不加, p 只是一个指代的, 代表了只能选择以下工作模式的其中一种:

<!-- TODO -->
* `d`       : 删除, delete modules from the archive, 要被删除的 modules 被指定为 member
* `r`       : insert with `r`eplacement, 将 member 插入到 archive. 
  * 之前存在的同名 member 会被替换
  * 如果某一个 member 文件不存在, 那么会中止并回退整个 ar 的运行结果
  * 默认会将 new members 插入到末尾, 可以通过 `mod` 来更改
  * `v` 会打印 archive 的详细运行信息
* `q`       : Quick append. 
  * 不会检测 exist, 并且总是会把 member 插入到 archive 的末尾
  * 会更新 list table 
  * `qs` 相当于 `r` 的别称
* `s`       : 更新 index, 该 operation 是个特例可以和其他的 operation 一起使用, 即本质上是一个可以独立运行的 `mod`


附加的 `[mod]` 代表了在主要的工作模式后, 一些附加的工作选项  

* `c`       : `create`, 会指定创建 archive 当这个 archive 不存在的时候. 如果不指定该 mod, 则 archive 不存在的时候 ar 会发出警告
* `s`       : 更新或者重建 一个 archive 的 `object-file index`, 即使 archive 没有发生任何改变. 可以不需要任何 operation 独立的运行 `ar s`, 这种情况下和另一个程序 `ranlib` 的功能相同


## 2.2. strings - 打印文件字符  



# 3. ld - the GNU linker.

GNU linker ld (GNU Binutils) version 2.40. 

ld 用于组合多个目标文件和 archive 文件, 重新定位它们的数据并绑定符号引用, 通常一个编译的最后一步过程就是 ld

ld 接受链接器注释语言 (Linker Command Language) files written in a superset of AT&T’s Link Editor Command Language syntax, 用于提供精细的以及完全的链接过程控制  
* 接受 基于 AT&T 链接编辑器命令 的超集 的链接器命令语言文件, 实现对链接过程的完全控制
* 使用通用库 `BFD libraries` 来操作目标文件, 这使得 ld 可以读取, 组合, 写入多种不同格式的目标文件, 例如 `COFF`? 和 a.out, 不同的格式可以链接在一起生成任何可用类型的目标文件
* ld 有高级报错信息 (作者吹了一波功能)


## 3.1. Invocation - 命令行工具使用说明

Command-line Options 除了全局的以外,  每一个平台似乎都有自己专有的命令. 不过看起来除了 i386 以外其他的都没多少内容
* Options Specific to i386 PE Targets
* Options specific to C6X uClinux targets
* Options specific to C-SKY targets
* Options specific to Motorola 68HC11 and 68HC12 targets
* Options specific to Motorola 68K target
* Options specific to MIPS targets
* Options specific to PDP11 targets


### 3.1.1. 通用链接命令




#### 3.1.1.1. 路径搜索配置

`--sysroot=director` : 指定某个路径为 `sysroot`
* 在其他命令中会代替 `$SYSROOT` 为前缀的路径
* 只有在 `--with-sysroot` 命令也出现的时候才会生效
* (文档里没有 `--with-sysroot` 的单独说明, 似乎只是一个开关 flag)


`-L searchdir` `--library-path=searchdir` : 指定一个目录作为库的搜索目录, 
* 该命令可以使用任意次, 按照顺序进行搜索, 且优先于所有默认内置的目录.  
* 所有 `-L` 指定的目录都会应用到 `-l` 命令上
* `-L` 不会影响到 ld 搜索链接描述文件的方式 (除非使用了`-T`选项)
* 路径可以使用 `=` 符号 或者 `$SYSROOT` 作为开头, 会替换为 `--sysroot` 选项指定的前缀
* 此外还可以使用 linker 脚本的 `SEARCH_DIR` 命令 


`-rpath-link=dir`   : 当链接的库本身依赖于其他的没有显式指定的库时, 用于操纵 ld 的行为. 在文档中该部分该详细解释了 ld 寻找共享库的顺序.
* `rpath-link=dir` 用于修改最高优先级的搜索路径, 该指令可以使用一些 tokens `${ORIGIN} ${LIB}`, 详细的优先级为
  * `-rpath-link` 指定的目录有最高优先级
  * (似乎不常见)`-rpath` 会在其后被参照, 差别在于 `-rpath-link` 仅在链接时使用, 而 `-rpath` 则似乎会被编译到可执行文件中, 因此会在运行时使用. 似乎只有应用了 `--with-sysroot` 选项的链接器才支持该搜索.
  * 在 ELF 系统中, 如果上述两个都未指定, 则查找环境变量 `LD_RUN_PATH`
  * 在 SunOS (Unix家族), 查找 `-L` 命令指定的路径
  * 查找环境变量 `LD_LIBRARY_PATH` 的目录
  * 在 ELF 系统中 查找环境变量 `DT_RUNPATH` or `DT_RPATH`, 前者的存在的话, 会阻止后者生效
  * 在 Linux 中, 查找 `/etc/ld.so.conf` 中定义的目录条目
  * 在 FreeBSD 系统上, 有 `elf-hints.h` 头文件定义的 `_PATH_ELF_HINTS` 宏指定的路径 (好复杂)
  * 参照 "linker script given on the command line", 命令中写的脚本中, 指定的 `SEARCH_DIR`, 不包括 `-dT` 的脚本
  * 默认链接目录, 通常为 `/lib` `/usr/lib`,  (没有 `/usr/local/lib`)
  * 由 Plugin 指定的路径 `LDPT_SET_EXTRA_LIBRARY_PATH`
  * 在 "default linker script" 中指定的 SEARCH_DIR

Note however on Linux based systems there is an additional caveat: If the --as-needed option is active and a shared library is located which would normally satisfy the search and this library does not have DT_NEEDED tag for libc.so and there is a shared library later on in the set of search directories which also satisfies the search and this second shared library does have a DT_NEEDED tag for libc.so then the second library will be selected instead of the first. 
* 啥意思, 对于系统上存在的多个 `libc.so` 会根据 `DT_NEEDED` 和 `--as-needed` flag 来优先选择有标志的 libc

总体看下来 `-L` 和 `/etc/ld.so.conf` 比较便于使用, LD_LIBRARY_PATH 写入到 `.bashrc` 也不错


### 3.1.2. Options Specific to i386 PE Targets - 主要的 PC 环境

i386 指代 32位x86架构的处理器, 即 Intel 80386 以及其后的32位处理器   
PE 指的是 Windows(为啥??), 全称为 Portable Executable.  

### 3.1.3. Environment Variables - ld 参照的环境变量

该章节指的环境变量是 ld 程序全局行为的环境变量, 而不是链接过程中的变量  
看起来都不常用  
* `GNUTARGET`
* `LDEMULATION`
* `COLLECT_NO_DEMANGLE`


## 3.2. BFD 库

ld linker 使用 BFD 库来访问以及打包目标文件.

# 4. GNU gprof - Displays profiling information.

Profiling a Program: Where Does It Spend Its Time?

gprof 由 Jay Fenlason 书写, 可以:
* 测定程序的哪一个部分最为消耗时间
* 测定某一个函数被调用了多少次


gprof 的使用方法包括:
* You must compile and link your program with profiling enabled.
  * 在编译和链接的时候加入 `-pg` 编译器参数
  * 如果使用 ld 来进行链接而非编译器的化, 需要指定 `gcrt0.o` as the first input file instead of the usual startup file `crt0.o`, 同时需要将 C library 的库文件由 `libc.a` 替换为 `libc_p.a`, 用于提供对标准库函数的调用次数统计支持
* You must execute your program to generate a profile data file. 
* You must run gprof to analyze the profile data. 


# 5. GNU Gprofng - Next Generation Profiler


