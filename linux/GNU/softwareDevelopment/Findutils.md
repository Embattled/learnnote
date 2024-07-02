# 1. GNU Findutils

Operating on files matching given criteria.

GNU Find Utilities, 是一个基本的路径搜索工具集. 

The tools supplied with this package are:
* find - search for files in a directory hierarchy
  * 用于在某个文件夹下查找特定目录, 并告知用户查找文件的出现路径
* locate - list files in databases that match a pattern
  * 扫描一个或者多个文件名数据库并显示文件的匹配项
  * 由于是基于一个本地的数据库上进行的查找因此非常快
* updatedb - update a file name database
  * 用于更新 locate 的数据库
* xargs - build and execute command lines from standard input
  * 通过收集在标准输入上读取的参数来构建命令行
  * 通常情况下都是由 find 生成的文件列表来生成


The manual is available in the `info system`` of the GNU Operating System. Use info to access the top level info page. Use info find to access the Find Utilities section directly. 

在线的可能更新不太即时的文档: https://www.gnu.org/software/findutils/manual/find.html


# 2. Introduction

基于 `find locate xargs` 三个命令的文件查找以及命令自动生成系统  

## 2.1. scope

在 Findutils 的概念里, `file` 代表 普通文件, 目录, symbolic-link, 以及其他实体  

`directory tree` 则是代表了一个目录以及其下面的所有文件和目录, 但也可以指代一个普通文件  

## 2.2. Overview


find 的基本语法 : `find [file…] [expression]`
* e.g. `find /usr/src -name '*.c' -size +100k -print`
* 通配符必须在引号内, 以防止它被 shell 解释

locate : `locate [option…] pattern…`
* e.g. `locate '*[Mm]akefile'`

xargs : `xargs [option…] [command [initial-arguments]]`
* e.g. `xargs grep typedef < file-list` :  从 file-list 中检测所有字符行, 并输出带有 typedef 的行


# Actions - 查找结果的使用方法

find 语句的查找结果可以直接被用来执行命令, 通过 find 本身的程序参数.

默认的行为就是打印所有查找结果

## Print File Name

* `-print`          : 在 std out 打印所有查找结果, 每一个结果追加一个 newline, 即一行一个
  * `-print0` : 结果不追加换行
* `-fprint file`    : 结果打印到文件中, 不管是否查找结果项目为空, 文件总是创建
  * `-fprint0 ` 不追加换行


## Print File Information

打印文件的详细信息

* `-ls` 

* `-printf format`

## Run Commands

将查找的结果用于命令行参数


### Single File

最基础的用法, 生成顺序的命令流, 同一时间只执行一个命令

* `-execdir command` : 执行命令
  * Execute command, 会根据 command 命令的执行结果, 如果为 0返回 true
  * `-execdir` 之后的所有参数都会作为 command 的项目, 直到 `;`
  * command 里的 `{}` 会被替换为当前查找到的文件名
  * 注意, 无论是 `{}` 或者 `;` 都需要通过 `\` 或者用引号括起来来避免在 shell 中直接被转义

`find . -name '*.h' -execdir diff -u '{}' /tmp/master ';'`


### Multiple Files

对于查找到的多个文件结果, 尽可能多的并行的执行多个命令, 节省时间



# 3. Reference

Below are summaries of the command line syntax for the programs discussed in this manual. 

程序本身的简要指令说明 
* find

## 3.1. Invoking find

完整的语法    

`find [-H] [-L] [-P] [-D debugoptions] [-Olevel] [file…] [expression]`

GNU程序都有的管理用程序  
* --help
    Print a summary of the command line usage and exit. 
* --version
    Print the version number of find and exit. 

### Filesystem Traversal Options

用于指定文件查找

### Find Expressions

用于指定表达式, 如果未指定, 则打印所有查找结果, 即默认使用 `-print`

该部分的完整命令在第三章 Action 中说明



