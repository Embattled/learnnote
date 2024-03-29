# GDB: The GNU Project Debugger

GDB, the GNU Project debugger, allows you to see what is going on inside another program while it executes -- or what another program was doing at the moment it crashed. 

https://www.sourceware.org/gdb/

GDB 的主要功能包括:
* 运行程序, 显示所有可能影响程序结果的信息
* 暂停程序在任何可能的地方
* 在程序暂停的时候检查发生的事件
* 在程序运行的时候更改内容, 使得可以验证其他的东西

GDB supports the following languages (in alphabetical order):
    Ada
    Assembly
    C
    C++
    D
    Fortran
    Go
    Objective-C
    OpenCL
    Modula-2
    Pascal
    Rust



# GDB Commands

GDB 的命令行构成

## Command Syntax
<!-- Over -->
同大多数 linux 程序一样, GDB command is a single line of input.

对于命令的缩写, 有一些命令会被赋予主要的缩写地位. 例如 `s` 会作为 `step` 的缩写, 尽管有很多其他以 s 开头的命令, 可以通过 `help` 命令来现场的测试一个 缩写指代的什么  

一个空输入 (直接按回车) 代表重复执行上一条输入的命令, 这不会对 `run` 生效.  如果盲目的按回车可能会导致麻烦, 因此这是一个可以被用户手动关闭的 feature

`#` 被用作 adb 命令的注释, 这主要被用在用脚本使用 adb 的情况下  

The `Ctrl-o` binding is useful for repeating a complex sequence of commands. This command accepts the current line, like RET, and then fetches the next line relative to the current line from the history for editing. 

## Command Settings


# backup

# 1. gdb 调试

要想使用完整gdb功能, 需要在编译的时候保存调试信息  
`g++ -g -o hello hello.cpp `


在命令行输入 `gdb` 可以直接进入程序  
`(gdb)` 开头的命令行交互  


基础命令：  
| **命令        | 解释                                               | 示例             |
| ------------- | -------------------------------------------------- | ---------------- |
| file <文件名> | 加载被调试的可执行程序文件                         | (gdb) file hello |
| r             | Run的简写，从头运行被调试的程序, 停在第一个断点    | (gdb) r          |
| c             | Continue的简写，从断点继续                         | (gdb) c          |
| q             | Quit的简写，退出GDB调试环境。                      | (gdb) q          |
| i             | info的简写，用于显示各类信息，详情请查阅“help i”。 | (gdb) i          |


断点设置与调试
* b <行号> b <函数名称> b <函数名称> b <代码地址>
  * b: Breakpoint的简写，设置断点。两可以使用“行号”“函数名称”“执行地址”等方式指定断点位置。 其中在函数名称前面加“*”符号表示将断点设置在“由编译器生成的prolog代码处”。如果不了解汇编，可以不予理会此用法。
  * (gdb) b 8 (gdb) b main (gdb) b *main (gdb) b *0x804835c 
* d [编号]
  * d: Delete breakpoint的简写，删除指定编号的某个断点，或删除所有断点。断点编号从1开始递增。                             
  * (gdb) d 
* s           
  * 执行一行源程序代码，如果此行代码中有函数调用，则进入该函数
  * s 相当于其它调试器中的“Step Into (单步跟踪进入)”
  * s 与 n 命令必须在有源代码调试信息的情况下才可以使用（GCC编译时使用“-g”参数）
  * (gdb) s
* n
  * 执行一行源程序代码，此行代码中的函数调用也一并执行。相当于其它调试器中的“Step Over (单步跟踪)”
  * (gdb) n 
* si, ni
  * si命令类似于s命令，ni命令类似于n命令。所不同的是，这两个命令（si/ni）所针对的是汇编指令
  * 而s/n针对的是源代码。
  * * (gdb) si (gdb) ni                            
* p <变量名称>Print的简写，显示指定变量（临时变量或全局变量）的值。
  * (gdb) p i 
  * (gdb) p nGlobalVar                           
* display … undisplay <编号>
  * 设置程序中断后欲显示的数据及其格式。 例如，如果希望每次程序中断后可以看到即将被执行的下一条汇编指令，可以使用命令 “display /i pc”其中pc 代表当前汇编指令，/i 表示以十六进行显示。当需要关心汇编代码时，此命令相当有用。 undispaly，取消先前的display设置，编号从1开始递增。
  * (gdb) display /i $pc (gdb) undisplay 1                 



## 1.1. 启动
```
$ gdb main.out -silent
```
选项`-silent`用于屏蔽 GDB 的前导信息，否则它会在屏幕上打印一堆免责条款  

## 1.2. 设置断点

在 GDB 中，设置断点的方法很多，包括在指定的内存地址处设置断点、在源代码的某一行设置断点，或者在某个函数的入口处设置断点  

```shell
#用内存地址的方式来设置这个断点
# 星号*意味着是以内存地址作为断点的。
b * 0x4004f4

#如果用源代码行的形式设置这个断点，则可以是
b 5
```

## 1.3. 运行

一旦设置了断点，下一步就是用`r`或者`run`命令执行被调试的程序，执行后会自动在第一个断点处停下来： 
```
[New Thread 1500.0x1e34]
[New Thread 1500.0x2fb8]
Thread 1 hit Breakpoint 1, main () at main.c:5
5     n = 1;
```
在内容的最后会显示下一条执行的语句以及内容  

## 1.4. 打印变量
```
(gdb) p n
$1 = 24
(gdb) p sum
$2 = 140737488347344
```
GDB 先计算表达式的值，并把它保存在一个存储区中，存储区的名字用$外加数字来表  

