# 1. GNU Bash

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


https://www.gnu.org/software/bash/

## 1.1. What is Bash? 
<!-- 完 -->
Bash 是一个 shell 或者可以说是 command language interpreter, 用于 GNUoperating system. 名称是 Bourne-Again SHell 的缩写.  Stephen Bourne 是当前 Unix shell `sh` 的直接祖先的作者, sh 出现在 Unix 的第七版贝尔实验室研究版本中.  

Bash 在很大程度上与 sh 兼容的, 并且 结合了 Korn shell(ksh) 以及 C shell (csh) 的优点.  
Bash 旨在提供一个 IEEE POSIX Shell and Tools 规范的实现 (IEEE Standard 1003.1).  

在 编程 和 交互 的方面提供了比 sh 更好的功能改进, 同时对于大多数 sh 程序都可以无需更改的使用 Bash 来运行.  

通常 GNU 操作系统可能会提供其他的 shell 包括 `csh`, 但普遍的 Bash 会是默认的 shell.  同其他的 GNU 软件一样, Bash 也拥有很强大的可移植性, 因此在 Unix 的每个版本以及其他的操作系统运行.  对于 MS-DOS, OS/2, Windows 等平台都存在独立支持的端口.  

## 1.2. What is shell?


# Definitions

These definitions are used throughout the remainder of this manual.   
文档中的固有名称

# 2. Basic Shell Features

Bash 是 `Bourne-Again SHell` 的缩写, 而 `Bourne shell` 就是最原始的由 `Stephen Bourne` 编写的 Shell

shell 的内置命令都被集成在了 bash 中, 而 evaluation 和 quoting 的规则都遵循了标准 Unix shell 的 POSIX 规范


该章节会介绍 shell 的 `building blocks`
* commands
* control structures
* shell function
* shell parameters
* shell expansions
* redirections (直接输入和输出命名文件的方法)
* how shell executes commands
<!-- 头部完 -->

## 2.1. Shell Syntax

## Shell Commands

shell 命令的介绍章节, shell 脚本编程的基础内容


### Reserved Words 保留字

用于 shell 脚本的流程控制  

```sh
if	then	elif	else	fi	time
for	in	until	while	do	done
case	esac	coproc	select	function
{	}	[[	]]	!

```
### Simple Commands
<!-- 完 -->
由一系列空格分隔的单词构成, 由 shell 的一个控制运算符 `control operators` 终止(?)

第一个单词用于指定要执行的命令, 跟随的单词用于程序的参数  

simple commands 命令的 return status 是由 POSIX 1003.1 waitpid 函数提供的退出状态  
如果命令由 `signal n` 终止而不是正常退出, 则返回值是 `128 + n`


### Pipelines

shell 管线, 通过 `|` 或者 `|&` 分割的多个命令  

`[time [-p]] [!] command1 [ | or |& command2 ] …`


### Lists of Commands

### Compound Commands

## Shell Functions

## Shell Parameters

## Shell Expansions

## Redirections



# 3. Shell Builtin Commands

所谓的 builtin commands 即不是由来于其他任何软件包, 而是 shell 本身所提供的 commands. 对于那些用单独的程序来实现几乎不可能或者不方便的功能来说, 内置 commands 是必须的. 

文档中 Builtin commands 包括了
* 从 `Bourne Shell` 中继承的命令
* 由 Bash 提供的或者拓展的命令


有一些其他的较为复杂的命令在文档中由其他章节来表述:
* Job control facilities
* directory stack
* command history
* programmable completion facilities

## 3.1. Bourne Shell Builtins

## 3.2. Bash Builtin Commands

## 3.3. Modifying Shell Behavior

自定义 Shell 的行为

### 3.3.1. The Set Builtin


# 4. Shell Variables

# 5. Bash Features

独立于 shell 的 Bash 独有的 feature
This chapter describes features unique to Bash. 

## 5.1. Invoking Bash

## 5.2. Bash Startup Files

## 5.3. Bash Conditional Expressions

bash 的条件判定表达式 

## 5.4. Shell Arithmetic



# 6. Job Control

介绍 Bash 提供的 任务管理功能

工作管理指的是在单个登录终端 也就是登录的 Shell 界面 同时管理多个工作的行为.  
把命令放入后台,然后把命令恢复到前台,或者让命令恢复到后台执行, 这些管理操作就是工作管理

后台管理有几个事项:
1. 前台是指当前可以操控和执行命令的这个操作环境, 后台是指工作可以自行运行, 但是不能直接用 `Ctrl+C` 快捷键来中止它, 只能使用 `fg/bg` 来调用工作
2. 当前的登录终端只能管理当前终端的工作,而不能管理其他登录终端的工作比如 tty1 登录的终端是不能管理 tty2 终端中的工作的
3. 放入后台的命令必须可以持续运行一段时间, 这样我们才能捕捉和操作它
4. 放入后台执行的命令不能和前台用户有交互或需要前台输入, 否则只能放入后台暂停, 而不能执行
   1. 比如 vi 命令只能放入后台暂停, 而不能执行, 因为 vi 命令需要前台输入信息, 
   2. top 命令也不能放入后台执行, 而只能放入后台暂停, 因为 top 命令需要和前台交互


常用的 nohup 命令不属于 Bash 的 job control 内置功能.  

## 6.1. Job Control Basics


工作控制即 用户可以有选择性的 停止/挂起/(suspend) 某一个 进程, 并在之后恢复其运行. 实现该功能主要依靠系统内核的驱动以及 Bash

shell 会把每一个 job 关联到每一个 pipeline. shell 会保存一个 job 表, 通过命令 `jobs` 可以输出在当前终端运行的工作表.  
当通过异步的方式启动一个 job 的时候, 会输出 一个数字以及一个 PID
* 数字是对于 shell 的该 job 的作业序号
* pid 则针对与该 job 关联的 pipeline 中, 多个进程中的最后一个进程, 即单个 pipeline 中的所有进程都属于同一个 job, bash 使用 job 的抽象概念作为 job control 的基础


如果操作系统支持 job control, 则 Bash 支持通过组合键来进行 Job 控制
* `Ctrl+Z ` : 挂起 进程, 并将控制权返还给 Bash
* `Ctrl+y ` : 当进程需要从终端读取字符输入的时候挂起, 并立即将 控制权返还给 Bash

有多种方法 能够在 shell 中引用一个 job, 关键字符为 `%` , % 会引出一个 jobspec
* `%n`  : 通过一个数字指代 Job number
* `%%` `%+` `%` : 指代 current job, 即最后一个在前台执行过的 job, 或者最后一个在后台启动的 job
* `%-`  : 前一个 job, 即 current job 的上一个
* 在和 job 相关的输出时, 会通过 `+-` 符号标记 对应的 current previous
* `%其他字符` `%?其他字符`: 进行起始字符匹配, 带问号的则是包含匹配


## 6.2. Job Control Builtins

介绍具体的 Bash 内置的 job control 机能  
* `bg [jobspec...] ` : 恢复在后台挂起的进程 使其在后台 继续工作, 就类似于该 job 是以 `&` 启动的那样
  * 如果 未提供 jobspec, 则针对当前 job
  * 正常情况下返回 0, 除非
    * job control is not enabled
    * job control enabled, jobspec was not found
    * specifies a job that was started without job control

* `fg [jobspec...]` : 恢复在后台挂起的进程, 并使其作为当前 job 工作在前台
  * 如果 未提供 jobspec, 则针对当前 job
  * 非 0 返回值的情况和上述相同

* `jobs [-lnprs] [jobspec]`  `jobs -x command [arguments]`
  * 打印所有活动的 jobs, 可以用来查看**当前终端**放入后台的工作
  * 基础 options
    * -l 列出进程的 PID 号
    * -n 只列出上次发出通知后改变了状态的进程
    * -p 只列出进程的 group leader 的 PID 号
    * -r 只列出运行中的进程
    * -s 只列出已停止的进程

* `kill`
* `wait`
* `disown`
* `suspend`


## 6.3. Job Control Variables

bash 提供了一个全局变量 `auto_resume`

该变量用于控制 shell 如何与 user 以及 job control 交互


# 7. Command Line Editing
<!-- 头部完 -->
本章介绍 GNU command line editing interface.  是 GUN readline 库的上层实现。 

默认情况下启动 Bash 的时候会启动 Command line editing.  除非带上 `--noediting` 启动 Bash

同理 Bash 的内置命令 read 也支持在 `-e` 参属下使用 command line editing 功能  

## 7.1. Introduction to Line Editing
<!-- 完 -->
介绍了在 Line Editing 中的键盘绑定
* C- 代表 Control 组合键
* M- 代表 Meta组合键, 在键盘中一般标记为 Alt
  * 对于没有 Alt 的键盘可以使用 Esc 作为代替, 先键入 ESC 松开后再键入对应字符, 即可被识别成 Alt 组合键
* M-C-k 即三个按键的组合键

其他特殊按键
* `RET` : Return or Enter
* `LFD` : 没有改键的话, 使用 C-j 代替
* `DEL ESC SPC TAB`  常见按键

## 7.2. Readline Interaction
<!-- 头部完 -->
主要用于修正命令, 例如在交互式命令行中, 注意到长命令的靠前字符错误, Readline 提供了快速移动光标的功能  
对于光标位置, 无论在哪, 键入 RET 都会让 Bash 接收到整行命令, 即无需将光标移动到末尾再键入

### 7.2.1. Readline Bare Essentials - Readline 按键绑定基础
<!-- 完 -->

* C-b C-f 向左向右移动光标, 目前主要被方向键代替 
* Backspace    删除光标左侧字符               
* DEL C-d      删除光标当前位置的字符
* C-_ C-x C-u  删除一段, Undo last editing command, 但是实测效果不统一

### 7.2.2. Readline Movement Commands
<!-- 完 -->
* C-a   : 移动到行首, 比 Home 键方便很多
* C-e   : 移动到行尾, 比 End 键快
* M-f   : 向前移动一个 word
* M-b   : 向后移动一个 word
* C-l   : clear 的快捷键, 方便

### 7.2.3. Readline Killing Commands
<!-- 完 -->
Killing test 意思是从行中删除文本, 但是会将其保存到剪贴板中, 以供以后使用, 通常是用于快速更改命令再行中的位置.

kill-yank 是较古老的叫法, 目前是称为 cut-paste

当 kill 文本多次, 每次的文本都会被追加的保存到同一个 `kill-ring`, 在之后单次的 yank 时会拉取最近的单词内容

* C-k   : kill 从当前位置到行尾的所有文本
* C-w   : kill 从当前位置到单词的起始, 或者是上一个单词的起始, 可以跨越多个空格
* M-d   : kill 从当前位置 word 结尾, 或者是下一个 word 的结尾
* M-DEL : kill 从当前位置 word 开头, 或者是下一个 word 的开头

* C-y   : 拉回最近被 kill 的一个 text 到光标位置
* M-y   : Rotate the kill-ring, 即在当前光标位置循环滚动 kill-ring , 知道找到想要插入的内容


### 7.2.4. Readline Arguments

### 7.2.5. Searching for Commands in the History

两种模式用于在历史命令记录中进行搜索  
* incremental       : 在每次键入搜索字符的时候都启动搜索
  * C-r 启动向后搜索
  * C-s 启动向前搜索, 但是实测用不了
  * ESC和 C-j 终止搜索, 并将当前历史条目应用到当前行
  * C-g 结束搜索, 并不应用搜索结果
  * RET 直接执行搜索到的结果
  * 移动命令 终止搜索并应用搜索结果, 开始编辑


# 8. Using History Interactively

介绍如何在 Bash 中使用 GNU History Library interactively, 本章主要作为一个 user's guide.  对于更多关于 GNU History Library 的信息, 则需要参考对应的专门文档.  


