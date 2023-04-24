# GDB: The GNU Project Debugger

GDB, the GNU Project debugger, allows you to see what is going on inside another program while it executes -- or what another program was doing at the moment it crashed. 

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