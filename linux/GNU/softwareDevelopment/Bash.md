# GNU Bash

Bash is the GNU Project's shell—the Bourne Again SHell.

是由 GNU 基金会所管理的 sh-compatible shell.  

一般 sh 用于指代最原始的 shell 软件 (Bourne shell). Developed by Stephen Bourne at Bell Labs.

Bash 的改进主要有:
* command-line editing.  命令行编辑?
* unlimited size command history. 无限制的历史存储功能
* job control. 
* shell functions and aliases
* indexed arrays of unlimited size
* integer arithmetic in any base from two to sixty-four.




## What is Bash? 
<!-- 完 -->
Bash 是一个 shell 或者可以说是 command language interpreter, 用于 GNUoperating system. 名称是 Bourne-Again SHell 的缩写.  Stephen Bourne 是当前 Unix shell `sh` 的直接祖先的作者, sh 出现在 Unix 的第七版贝尔实验室研究版本中.  

Bash 在很大程度上与 sh 兼容的, 并且 结合了 Korn shell(ksh) 以及 C shell (csh) 的优点.  
Bash 旨在提供一个 IEEE POSIX Shell and Tools 规范的实现 (IEEE Standard 1003.1).  

在 编程 和 交互 的方面提供了比 sh 更好的功能改进, 同时对于大多数 sh 程序都可以无需更改的使用 Bash 来运行.  

通常 GNU 操作系统可能会提供其他的 shell 包括 `csh`, 但普遍的 Bash 会是默认的 shell.  同其他的 GNU 软件一样, Bash 也拥有很强大的可移植性, 因此在 Unix 的每个版本以及其他的操作系统运行.  对于 MS-DOS, OS/2, Windows 等平台都存在独立支持的端口.  

## What is shell?



# Shell Builtin Commands

所谓的 builtin commands 即不是由来于其他任何软件包, 而是 shell 本身所提供的 commands. 对于那些用单独的程序来实现几乎不可能或者不方便的功能来说, 内置 commands 是必须的. 

文档中 Builtin commands 包括了
* 从 `Bourne Shell` 中继承的命令
* 由 Bash 提供的或者拓展的命令


有一些其他的较为复杂的命令在文档中由其他章节来表述:
* Job control facilities
* directory stack
* command history
* programmable completion facilities


# Job Control

介绍 Bash 提供的 任务管理功能


# Using History Interactively

介绍如何在 Bash 中使用 GNU History Library interactively, 本章主要作为一个 user's guide.  对于更多关于 GNU History Library 的信息, 则需要参考对应的专门文档.  


