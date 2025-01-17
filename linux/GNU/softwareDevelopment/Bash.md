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


# 2. Definitions

These definitions are used throughout the remainder of this manual.   
文档中的固有名称

# 3. Basic Shell Features

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

## 3.1. Shell Syntax

## 3.2. Shell Commands - shell 命令

shell 命令的介绍章节, shell 脚本编程的基础内容



### 3.2.1. Reserved Words 保留字
<!-- 完 -->

用于 shell 脚本的流程控制  

相对来说没有那么多

```sh
if	then	elif	else	fi	time
for	in	until	while	do	done
case	esac	coproc	select	function
{	}	[[	]]	!

```
### 3.2.2. Simple Commands
<!-- 完 -->
由一系列空格分隔的单词构成, 由 shell 的一个控制运算符 `control operators` 终止(?)

第一个单词用于指定要执行的命令, 跟随的单词用于程序的参数  

simple commands 命令的 return status 是由 POSIX 1003.1 waitpid 函数提供的退出状态  
如果命令由 `signal n` 终止而不是正常退出, 则返回值是 `128 + n`


### 3.2.3. Pipelines - 管线
<!-- 完 -->
shell 管线, 通过 `|` 或者 `|&` 分割的多个命令  

`[time [-p]] [!] command1 [ | or |& command2 ] …`

前一个命令的输出会作为下一个命令的输入, 即每一个命令都会读取前一个命令的输出.
管线和命令输出的重定向是有前后优先顺序的  
* 标准管线会在 重定向之前构建链接
* `|&` 相比于普通管线, 标准错误信息流也会被关键输入到第二个命令中, 但是该隐式重定向会在 command1 的重定向之后执行
* 如果管线命令不是异步执行的话, shell 会挂起 直到命令执行完成
* 管线的多个命令都会在自己的子 shell 中执行, 除了最后一个命令.
  * 受 shopt lastpipe 选项 和 job control 影响  
* 管线命令的退出状态也是最后一个命令的退出状态.
  * 受 `pipelinefail` 选项影响, 会导致返回值是最后一个非 0 状态退出的命令  

在管线的格式中 可以加入 bash 保留字命令 `time [-p]` 用于统计整个管线多命令的执行时间 (带管线的执行很难通过外部命令来方便的计时)
* 包括 elapsed(wall-clock) 时间, 以及 命令执行过程中消耗的 用户 和 系统 时间
* `-p` 选项会将格式更改为 POSIX 指定的格式, 如果Bash 本身已经处于 POSIX 模式, 则 time 后接一个 `-` 的选项会导致 time 无法被识别为 保留字, 需要注意




### 3.2.4. Lists of Commands - 命令列表

通过运算符分割的多个 命令或者 管道命令

命令间的分割 运算符有 `;  & && ||`, 以 `; & newline` 结尾  
* `&& ||` 优先级最高
* `& ;` 其次
* lists of commands 中的 newline 相当于 `;`


以 `&` 结尾的命令 会启用异步执行模式, shell 立即返回 0, 命令会在后台执行.  
* 如果 Job Control 没有启动, 并且没有任何重定向的话, 异步执行的命令会从 `/dev/null` 读取输入 (即没有任何输入)

以 `;` 结尾的命令, 会由 shell 一次执行, 最终返回的是最后执行的命令的退出状态


以 `|| &&` 分割的命令代表 OR AND 控制, 以 左关联性执行  
* `command1 && command2` 只有 command1 返回 0, command2 才会执行
* `command1 || command2` 只有 command1 返回非0, command2 才会执行
* 最终的返回值是最后执行的命令的退出状态


### 3.2.5. Compound Commands - 组合命令
<!-- 头部完 -->
复合命令是 bash 编程的基础

每一个 compound 构造都是以 保留字 或者 控制运算符 开始, 并以相应的 保留字或者 运算符种植  

在大多数情况下, 复合命令中的命令可以通过 换行符 与 命令中的其他部分分割开, 并且后面可以用换行符号替换分号  

bash 提供了 3 中基本复合命令结构


#### Looping Constructs - 循环结构
<!-- 完 -->

bash 中支持 3 种循环结构, 语法中的 `;` 可以被替换为 newline
* until `until test-commands; do consequent-commands; done`
  * 在 `test-commands` 返回值不为 0 的时候执行循环里的命令
* while `while test-commands; do consequent-commands; done`
  * 在 `test-commands` 的返回值 为 0 的之后执行循环里的
* for
  * 遍历列表 `for name [ [in [words …] ] ; ] do commands; done`
    * `in words` 是可以被省略的, 相当于执行  `in "$@"`
    * 需要参照 Special Parameters
  * C风格的 for `for (( expr1 ; expr2 ; expr3 )) ; do commands ; done`

循环的内容一定是夹在  do done 中间的

保留字  `continue` 和  `break` 都可以用在 loop 循环中

* value_list 的形式有多种
  * 直接给出具体的值, 甚至不需要加括号, 也没有逗号  `in a b c d 1 2 3`
  * 给出一个取值范围, 需要大括号 `{start..end}`, 只支持数字和 ASCII 的字符
  * 使用命令的执行结果
  * 使用 Shell 通配符 
  * 使用特殊变量

```shell
# C 风格
for((exp1; exp2; exp3))
do
    statements
done

# python 风格
for variable in value_list
do
    statements
done


#!/bin/bash
sum=0
for ((i=1; i<=100; i++))
do
    ((sum += i))
done
echo "The sum is: $sum"
```


#### Conditional Constructs - 条件结构

shell 支持的几种条件语法
* `if then elif else fi`
* `case esac`
* `select do done`
* `(())`
* `[[]]`

* if else elif 语句
  * fi 用于结构闭合
  * else 用于最后的分支
  * elif 用于继续添加分支

```shell
# 如果 condition 成立, 那么 then 后边的语句将会被执行
# fi 必须存在, 用来闭合该选择结构
if  condition
then
    statement(s)
fi
# 带 else 的分支
if  condition
then
   statement1
elif condition3
then
    statement3
else
   statement2
fi
# 可以用分号 ; 来改变选择结构的代码风格
if  condition;  then
    statement(s)
fi

```
shell 也有对应的 switch 语句
* 关键字 `case in` 进入选择结构
* 关键字 `esac`    推出结构
* 对于每个 case
  * 使用 右括号来标识一个 case `)`
  * 使用一个 pattern`)` 来进行匹配, 可以是 数字, 字符串, 甚至是正则表达式
  * 使用双分号来结束 case `;;`  , 该符号相当于 C 语言中的 `break`, 属于语法要求, 必须要有
  * 最后放一个 `*)` 用来拖地, 相当于 default, 这个分支没有双分号也可以

特殊 pattern 总结
| 格式    | 说明                                                                                         |
| ------- | -------------------------------------------------------------------------------------------- |
| `*`     | 表示任意字符串。                                                                             |
| `[abc]` | 表示 a、b、c 三个字符中的任意一个。比如, `[15ZH]` 表示 1、5、Z、H 四个字符中的任意一个。     |
| `[m-n]` | 表示从 m 到 n 的任意一个字符。比如, `[0-9]` 表示任意一个数字, `[0-9a-zA-Z]` 表示字母或数字。 |
| `|`     | 表示多重选择, 类似逻辑运算中的或运算。比如, abc                                              | xyz 表示匹配字符串 "abc" 或者 "xyz" |


```shell
case expression in
    pattern1)
        statement1
        ;;
    pattern2)
        statement2
        ;;
    pattern3)
        statement3
        ;;
    ……
    *)
        statementn
esac
```

#### Grouping Commands - 分组命令
<!-- 完 -->
两种括号 用于将一部分命令化作整体执行, 这样 整体的输出可以被用于 管道 或者重定向  

* `(list)`
  * 圆括号里的 命令会被强制创建在一个 子 shell 中执行
  * 子 shell 中的变量不会保持有效性
  * 圆括号是运算符, 因此不需要和命令以空格分割
* `{list;}`
  * 分号是语法必须的
  * 花括号里的命令会在当前 shell 中被执行
  * 花括号 在 shell 中是保留字, 因此需要用空格与 列表中的命令分隔开


### Coprocesses - 协程

### GNU Parallel - GNU 并行执行

详细使用方法记录在了别的 软件文档下  

https://www.gnu.org/software/parallel/parallel_tutorial.html. 


## 3.3. Shell Functions

定义 shell 函数
* `fname () compound-command [ redirections ]`
* `function fname [()] compound-command [ redirections ]`


shell 的函数定义没有太大区别
* function 是关键字, 但是可以省略
* return 是可选的关键字, 只能返回一个值

shell 的函数调用比较简化
* 直接给出函数名字即可, 函数名字后面都不需要带括号
* 如果传递参数, 那么多个参数之间以空格分隔

```shell
function name() {
    statements
    [return value]
}

name() {
    statements
    [return value]
}

# 调用函数
name
# 带参数调用
name param1 param2 param3
```



## 3.4. Shell Parameters

### Positional Parameters - 位置参数

### Special Parameters - 特殊参数 (天书参数)

特殊变量种类不是很多

| 变量      | 含义                                                         |
| --------- | ------------------------------------------------------------ |
| $0        | 执行该脚本的 bash 名                                         |
| $n（n≥1） | 传递给脚本或函数的参数                                       |
| $#        | 传递给脚本或函数的参数个数                                   |
| $*        | 传递给脚本或函数的所有参数                                   |
| $@        | 传递给脚本或函数的所有参数。                                 |
| $?        | 上个命令的退出状态, 或函数的返回值                           |
| $$        | 当前 Shell 进程 ID 对于 Shell 脚本 就是这些脚本所在的进程 ID |

* Shell 位置参数  : 即运行 Shell 脚本文件时传递的命令行参数, 或者函数参数
  * 这些参数在脚本文件内部可以使用 `$n` 的形式来接收
  * shell 在定义函数时不用指明参数的名字和数目
  * Shell 函数参数没有所谓的形参和实参
  * 总结, shell 语言在定义参数时不能带参数, 但是在调用函数可以直接传递参数
  * 传入函数的参数也是直接用 `$n` 来接受


* `$*` 和 `$@` 的区别
  * 直接使用 (不使用双引号包括的时候)  没有区别, 参数间以空格分开
  * 使用双引号包围的时候:
  * `$*` 会将所有的参数从整体上看做一份数据
  * `$@` 仍然将每个参数都看作一份数据
  * 具体来说, `"$*"` 作为 for in 形式的数据表会导致不能区分各个参数 

* `$?` 用来获取上一个命令的退出状态, 或者上一个函数的返回值
  * 一般情况下, 大部分命令执行成功会返回 0
  * Shell 函数中的 return 关键字用来表示函数的退出状态


## 3.5. Shell Expansions

### Command Substitution - 命令替换
<!-- 完 -->
将命令的输出替换掉命令本身, 有两种语法
* `$(command)`
* `command`
命令会在 subshell 中执行, 并将命令的 stdout 替换为 该字符本身  


将命令的输出结果赋值给某个变量
* 获取命令的结果有两种方式, 反引号和括号
  * 而且有些情况必须使用 `$()`：因为`$()` 支持嵌套, 反引号不行
  * `$()` 仅在 Bash Shell 中有效
  * 反引号可在多种 Shell 中使用
  * 反引号毕竟看起来像单引号, 有时候会对查看代码造成困扰

* 如果被替换的命令的输出内容包括多行 (有换行) 或者含有多个连续的空白符
  * 系统会使用默认的空白符来填充, 导致换行无效
  * 连续的空白符默认被压缩成一个
  * 输出变量时将变量用双引号包围可以保证原本输出  
```shell
# 反引号和单引号容易混淆, 最好别用
variable=`command`
variable=$(command)
time=$(date)

LSL=`ls -l`
echo "$LSL"  #使用引号包围
# 可以有多个命令, 多个命令之间以分号;分隔


# 读取一个文件可以用 (< file) 执行更快
$(cat file) 
$(< file)
```

### Arithmetic Expansion - 数学计算

`$(( expression ))` 计算一个表达式的数值结果, 并进行内容替换  

数学表达式的 expansions 和 双引号 double quotes 大概相同
* 表达式中的 双引号字符不会被特殊处理, 并被删除
* 表达式中的所有 tokens 都会进行拓展, 包括
  * variable expansion
  * command substitution
  * quote removal

双小括号
* 双小括号 `((表达式))` 是 Bash Shell 中专门用来进行整数运算的命令
* 可以使用 `$` 获取 (( )) 命令的结果
* 在 `(( ))` 中使用变量无需加上 `$ `前缀

```shell
# 计算完成后给变量赋值
((a=10+66))
a=$((10+66)

# 对多个表达式同时进行计算
((a=3+5, b=a+10))

# 错误写法, 不加 $ 就不能取得表达式的结果
c=((a+b)) 


```

## 3.6. Redirections



# 4. Shell Builtin Commands

所谓的 builtin commands 即不是由来于其他任何软件包, 而是 shell 本身所提供的 commands. 对于那些用单独的程序来实现几乎不可能或者不方便的功能来说, 内置 commands 是必须的. 

文档中 Builtin commands 包括了
* 从 `Bourne Shell` 中继承的命令
* 由 Bash 提供的或者拓展的命令


有一些其他的较为复杂的命令在文档中由其他章节来表述:
* Job control facilities
* directory stack
* command history
* programmable completion facilities

## 4.1. Bourne Shell Builtins

从 Bourne Shell 中继承的功能, 按照 POSIX 标准来实现  



## 4.2. Bash Builtin Commands

bash 相比于 sh 中特有的命令, 其中一些是作为 POSIX 标准被引入 bash 里的
* read 读取一行并以某一个字符分割






read:  
```sh
read [-ers] [-a aname] [-d delim] [-i text] [-n nchars]
    [-N nchars] [-p prompt] [-t timeout] [-u fd] [name …]
```



## 4.3. Modifying Shell Behavior

自定义 Shell 的行为

### 4.3.1. The Set Builtin


# 5. Shell Variables

# 6. Bash Features

独立于 shell 的 Bash 独有的 feature
This chapter describes features unique to Bash. 

## 6.1. Invoking Bash

## 6.2. Bash Startup Files

## 6.3. Bash Conditional Expressions

bash 的条件判定表达式 

## 6.4. Shell Arithmetic



# 7. Job Control

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

## 7.1. Job Control Basics


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


## 7.2. Job Control Builtins

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


## 7.3. Job Control Variables

bash 提供了一个全局变量 `auto_resume`

该变量用于控制 shell 如何与 user 以及 job control 交互


# 8. Command Line Editing
<!-- 头部完 -->
本章介绍 GNU command line editing interface.  是 GUN readline 库的上层实现。 

默认情况下启动 Bash 的时候会启动 Command line editing.  除非带上 `--noediting` 启动 Bash

同理 Bash 的内置命令 read 也支持在 `-e` 参属下使用 command line editing 功能  

## 8.1. Introduction to Line Editing
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

## 8.2. Readline Interaction
<!-- 头部完 -->
主要用于修正命令, 例如在交互式命令行中, 注意到长命令的靠前字符错误, Readline 提供了快速移动光标的功能  
对于光标位置, 无论在哪, 键入 RET 都会让 Bash 接收到整行命令, 即无需将光标移动到末尾再键入

### 8.2.1. Readline Bare Essentials - Readline 按键绑定基础
<!-- 完 -->

* C-b C-f 向左向右移动光标, 目前主要被方向键代替 
* Backspace    删除光标左侧字符               
* DEL C-d      删除光标当前位置的字符
* C-_ C-x C-u  删除一段, Undo last editing command, 但是实测效果不统一

### 8.2.2. Readline Movement Commands
<!-- 完 -->
* C-a   : 移动到行首, 比 Home 键方便很多
* C-e   : 移动到行尾, 比 End 键快
* M-f   : 向前移动一个 word
* M-b   : 向后移动一个 word
* C-l   : clear 的快捷键, 方便

### 8.2.3. Readline Killing Commands
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


### 8.2.4. Readline Arguments

### 8.2.5. Searching for Commands in the History

两种模式用于在历史命令记录中进行搜索  
* incremental       : 在每次键入搜索字符的时候都启动搜索
  * C-r 启动向后搜索
  * C-s 启动向前搜索, 但是实测用不了
  * ESC和 C-j 终止搜索, 并将当前历史条目应用到当前行
  * C-g 结束搜索, 并不应用搜索结果
  * RET 直接执行搜索到的结果
  * 移动命令 终止搜索并应用搜索结果, 开始编辑


# 9. Using History Interactively

介绍如何在 Bash 中使用 GNU History Library interactively, 本章主要作为一个 user's guide.  

对于更多关于 GNU History Library 的信息, 例如使用 History Library 进行软件开发, 则需要参考对应的 `Readline Library` 专门文档.  


## Bash History Facilities

## Bash History Builtins

Bash 提供了 2 个内置组件用于操纵历史记录  
* fc
* history


```sh
fc [-e ename] [-lnr] [first] [last]
fc -s [pat=rep] [command]
```



## History Expansion



