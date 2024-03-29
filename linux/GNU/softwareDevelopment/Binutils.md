- [1. GNU Binutils](#1-gnu-binutils)
- [2. GNU binary utilities (binutils)](#2-gnu-binary-utilities-binutils)
  - [2.1. ar  (archive)](#21-ar--archive)
    - [2.1.1. operation keyletter](#211-operation-keyletter)
  - [strings - 打印文件字符](#strings---打印文件字符)
- [3. ld - the GNU linker.](#3-ld---the-gnu-linker)
- [4. GNU gprof - Displays profiling information.](#4-gnu-gprof---displays-profiling-information)
- [GNU Gprofng - Next Generation Profiler](#gnu-gprofng---next-generation-profiler)

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


## strings - 打印文件字符  



# 3. ld - the GNU linker.

GNU linker ld (GNU Binutils) version 2.40. 

ld 用于组合多个目标文件和 archive 文件, 重新定位它们的数据并绑定符号引用, 通常一个编译的最后一步过程就是 ld

ld 接受链接器注释语言 (Linker Command Language) files written in a superset of AT&T’s Link Editor Command Language syntax, 用于提供精细的以及完全的链接过程控制  



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


# GNU Gprofng - Next Generation Profiler


