# 1. make

对于任何编译型语言首
1. 先要把源文件编译成中间代码文件, Windows下即 `.obj` 文件, UNIX下是 `.o` 文件 Object File, 这个动作叫做编译 (compile)
   * 一般来说, 每个源文件都应该对应于一个中间目标文件 (O文件或是OBJ文件) 。 
   * 如果将中间目标文件打个包, 在Windows下这种包叫`库文件`  (Library File), 也就是 .lib 文件, 在UNIX下, 是Archive File, 也就是 .a 文件
2. 然后再把大量的Object File合成执行文件, 这个动作叫作链接 (link) 


对于G++来说如果不用 makefile, 则需要按照下面的方式编译上述代码: 
```
g++ -c function1.cpp
g++ -c function2.cpp
g++ -c main.cpp
g++ -o hello main.o function1.o function2.o
```

make 的作用就是自动化管理以上过程, 并自动跳过已经编译过且没有新的更新的文件  


在该目录下直接输入命令 `make` 来使用  
makefile文件最好直接命名为`makefile`或`Makefile`

当然可以使用别的文件名来书写Makefile, 比如: `Make.Linux` , `Make.Solaris` , `Make.AIX` 等  

如果要指定特定的Makefile, 你可以使用make的 -f 和 --file 参数, 如:  `make -f Make.Linux` 或 `make --file Make.AIX`


# 2. make CLI

* `-j [N], --jobs[=N]`
    * Allow N jobs at once; infineite jobs with no arg.
    * 允许并行编译
    * 不指定数字的时候自动探测最大的可用CPU
* `-f, --file` 
    * 当 makefile 文件不是默认的时候, 用该参数来指定要执行的 makefile 文件

# 3. makefile


makefile 可以简单理解为更高级的 shell 脚本, 具有一些检测功能, 一个 makefile 可以按照以下 5 种部件来构成
* explicit rules
* implicit rules
* variable fefinitions
* directives
* comments              : 注释




* target        : 为一个标签 label, 在实际中可以具体指 目标文件或者执行文件, 或者是单纯的命令集(伪目标)
  * 命令行输入 `make` 将默认执行**第一个** `target`  (即 all) 下方的命令  
  * 如果输入 `make 关键字`, 则会直接跳转到对应 target 的执行部分
  * 如果某个 target 没有被 `第一个 target` 所直接/间接管理, 那么这个 target 的部分则是可选的, 不会被自动执行, 只能通过 `make 关键字` 来手动执行, 一般都是利用该特性来定义 `make clean`
* prerequisites : 是当 label 是一个目标文件时, 要生成该 target 所需要的源文件. 当 label 是可执行文件的时候, 所需要的 目标文件 
* command       : 具体的 shell 命令, 定义了如何生成目标文件的操作系统命令, 一定要以一个 Tab 键作为开头  

所谓 make 提供的, 就是一系列的规则:
* 例如自动关键字替换



## 3.1. 基础 makefile 


```makefile
all:
        g++ -o hello main.cpp function1.cpp function2.cpp
clean:
        rm -rf *.o hello
```
1. 无 prerequisites 利用的 makefile :
   * 可以省去敲命令的痛苦
   * 无法选择性编译源码. 因为所有源文件的编译和链接都在同一条 command 里, 导致每次都要编译整个工程, 很浪费时间


```makefile
all: hello
hello: main.o function1.o function2.o
        g++ main.o function1.o function2.o -o hello
main.o: main.cpp
        g++ -c main.cpp
function1.o: function1.cpp
        g++ -c function1.cpp
function2.o: function2.cpp
        g++ -c function2.cpp

clean:
        rm -rf *.o hello
```

2. 通过指定 prerequisites 来减少冗余编译
   1. 命令行输入 make, 将默认执行 all 这个 target
   2. all 这个 target 依赖于 hello, hello 在当前目录下并不存在, 于是程序开始往下读取命令, 定位到 `hello` 这个 `target`
   3. 正待执行 `hello` 这个 target 的时候, 却发现它依赖于 `main.o`, `function1.o`, `function2.o` 这三个文件, 而它们在当前目录下都不存在, 于是程序继续向下执行
   4. 遇到 main.o target, 它依赖于 main.cpp。而 main.cpp 是当前目录下存在的文件, 终于可以编译了, 生成 main.o 对象文件。后面两个函数以此类推, 都编译好之后, 再回到 hello target, 连接各种二进制文件, 生成 hello 文件。



## 4.2. 特殊符号

可以极大的简化 makefile 的书写, 同时极大的提高 makefile 的阅读门槛

### 指代符号


对于存在的一个段 `all: library.cpp main.cpp`  
* `$@` : 本段的`target`  即 `all`
* `$<` : `library.cpp`   即第一个 dependency
* `$^` : library.cpp 和 main.cpp  即**所有的** dependencies

我们的 dependencies 中的内容, 往往和 g++ 命令中的内容重复,例如: 
```makefile
hello: main.o function1.o function2.o
        $(CC) $(LFLAGS) main.o function1.o function2.o -o hello

# 使用特殊符号简化后
hello: main.o function1.o function2.o
        $(CC) $(LFLAGS) $^ -o $@        
```

## 4.3. 自动检测目录
自动检测目录下所有的 cpp 文件呢？此外 main.cpp 和 main.o 只差一个后缀, 能不能自动生成对象文件的名字, 将其设置为源文件名字后缀换成 .o 的形式

`wildcard` 用于获取符合特定规则的文件名  
```makefile
SOURCE_DIR = . # 如果是当前目录, 也可以不指定
SOURCE_FILE = $(wildcard $(SOURCE_DIR)/*.cpp)
#  $(wildcard *.cpp))
target:
    # 输出的为当前目录下所有的 .cpp 文件
    @echo $(SOURCE_FILE)
```
其中 @echo 前加 @是为了避免命令回显, 上文中 make clean 调用了 `rm -rf` 会在 terminal 中输出这行命令, 如果在 `rm` 前加了 `@` 则不会输出了  

`patsubst`用它可以方便地将 .cpp 文件的后缀换成 .o  
它的基本语法是: `$(patsubst 原模式, 目标模式, 文件列表) ` 
```makefile
SOURCES = main.cpp function1.cpp function2.cpp
# 目标文件都是 .o 结尾的, 那么就将其表示为 %.o
OBJS = $(patsubst %.cpp, %.o, $(SOURCES))
target:
        @echo $(SOURCES)
        @echo $(OBJS)
```

## 4.4. 自动推导(简化makefile代码)

Static Pattern Rule  
`targets: target-pattern: prereq-patterns`  
这里的targets代表一组target  例如  

`OBJS = $(patsubst %.cpp, %.o, $(wildcard *.cpp))`  


* target-pattern 代表target的特征
* prereq-patterns 代表原本的依赖文件的特征

最终通过一行可以执行全部的文件编译 
```makefile
$(OBJS):%.o:%.cpp
        $(CC) $(CFLAGS) $< -o $@

```

# 5. 高级操作

## 5.1. 多makefile处理

在Makefile使用 include 关键字可以把别的Makefile包含进来, 这很像C语言的 `#include`  

`include filename`
filename 可以是当前操作系统Shell的文件模式 (可以包含路径和通配符) 

多个filename用一个或多个空格隔开,例  
```makefile
# 想要包含的makefile文件有
# a.mk 、 b.mk 、 c.mk  foo.make 
# 一个变量 $(bar) , 其包含了 e.mk 和 f.mk

include foo.make *.mk $(bar)
#等价于
include foo.make a.mk b.mk c.mk e.mk f.mk
```

如果文件都没有指定绝对路径或是相对路径的话, make会在当前目录下首先寻找, 如果当前目录下没有找到, 那么, make还会在下面的几个目录下找: 
1. 如果make执行时, 有 -I 或 --include-dir 参数, 那么make就会在这个参数所指定的目录下去寻找。
2. 如果目录 <prefix>/include  (一般是:  `/usr/local/bin` 或 `/usr/include` ) 存在的话, make也会去找。

* 如果有文件没有找到的话, make会生成一条警告信息, 但不会马上出现致命错误。它会继续载入其它的文件
* 一旦完成makefile的读取, make会再重试这些没有找到, 或是不能读取的文件, 如果还是不行, make才会出现**一条致命信息**。
* 如果你想让make不理那些无法读取的文件, 而继续执行, 你可以在include前加一个减号`-` 。如: `-include <filename>`



# Rules

makefile 的规则 是以下的形式  
```makefile
# syntax
targets : prerequisites
        recipe
targets : prerequisites ; recipe
        recipe
        …

# example
foo.o : foo.c defs.h       # module for twiddling the frobs
        cc -c -g foo.c
```
`rule` 用于描述何时重新生成一个目标文件
* `prerequisites`中如果有一个以上的文件比`target`文件要新的话, command所定义的命令就会被执行, 否则就跳过该 command  
* make 执行的时候, 会在 shell 打印所执行的具体的 receipe (编译/链接命令)


一个 makefile 里面会有多条 rule , 其中大部分的规则不存在前后顺序, 除了被称为 `default goal` 的 rule:
* `default goal` 默认被定义为 makefile 中的第一条 rule
* 在 CLI 中键入 `make` 时所执行的生成目标
* 因为 rule 之间会有链式关联, 所以所谓 default goal 其实可以是多个 rule 的合集
* 作为例外:
  * 以点 `.` 开头的 target 不会成为 default goal
  * 不被当前的主 `default goal` 所间接直接依赖的 target 不是 `default goal`
* 作为一个通识, 编写 makefile 的时候会将 first rule 用于描述整个项目程序的生成, 该 target 经常写成 `all`

## Phony Target 

Phony Target 是一种特殊的目标, 他并不是一个真实的生成目标文件, 更像是一个 recipe 的集合的命名, 主要用于 make 执行具体的 request 的时候  
* to avoid a conflict with a file of the same name
* and to improve performance

```makefile
clean:
        rm *.o temp

.PHONY : clean
clean :
    -rm edit $(object)
```

对于常用的 clean rule:
* 由于其没有 prerequisites, 且他的 recipe 不会生成名为 clean 的文件, 所以 make 会在每次调用 `make clean` 的时候都执行 `rm` 命令
* 然而若是当前目录中存在了名为 `clean` 的任意文件, 则由于没有 prerequisites, 导致 make 认为 clean 已经是最新的, 从而导致相应的 rm 命令被跳过
* 解决办法就是, 将 clean rule 实现声明为 Phony Target


## Special Built-in Target Names

一些特殊的 name 作为 Target 的时候会有特殊的意思

像是 `clean`, 因为没有依赖文件,所以不会被自动执行,但有更稳妥的书写格式  


* .PHONY : prerequisites of the special target .PHONY are considered to be phony targets


```makefile
clean:
    rm edit $(objects)


```
而在 rm 命令前面加了一个小减号的意思就是, 也许某些文件出现问题, 但不要管, 继续做后面的事  

# Variables

makefile 的 variable 是一些被定义为 name 的 string or text, 这些 varable 的值可以替换在任何部分 (target, prerequisites, recipes, etc.)

在一些其他版本的 make 里, variables 也被叫做 macros

variables 的命令规则:
* 不能有 `: # =` 符号
* 以点 `.` 和 大写字母开头的 name 有可能会有特殊的意思
* variable 的命名区分大小写



## 4. makefile 变量

我们希望将需要反复输入的命令整合成变量, 用到它们时直接用对应的变量替代, 这样如果将来需要修改这些命令, 则直接修改变量的值即可

makefile 的变量使用类似于 shell , 即定义的时候使用 `=` 进行赋值, 具体进行变量值替换的时候使用 `$(变量名)`


```makefile
CC = g++
CFLAGS = -c -Wall # 编译同时显示错误
LFLAGS = -Wall # 链接同时显示错误

all: hello
hello: main.o function1.o function2.o
        $(CC) $(LFLAGS) main.o function1.o function2.o -o hello
main.o: main.cpp
        $(CC) $(CFLAGS) main.cpp
function1.o: function1.cpp
        $(CC) $(CFLAGS) function1.cpp
function2.o: function2.cpp
        $(CC) $(CFLAGS) function2.cpp

clean:
        rm -rf *.o hello
```
使用变量来定义 编译/链接的 FLAG 的 makefile
* -Wall 表示显示编译过程中遇到的所有 `warning`  
* 引用变量名时需要用 $() 将其括起来, 表示这是一个变量名  
* 这里的编译和链接都使用了 `-Wall`, 但将他们分别定义为 CFLAGS 和 LFLAGS


```makefile
objects = main.o kbd.o command.o display.o \
    insert.o search.o files.o utils.o
edit : $(objects)
    cc -o edit $(objects)
clean :
    rm edit $(objects)
```
在makefile中以 $(objects) 的方式来通过变量来管理目标文件   
如果有新的 .o 文件加入, 简单地修改一下 objects 变量就可以了


## Setting Variables 设置变量

To set a variable from the makefile, write a line starting with the variable name followed by one of the assignment operators` ‘=’, ‘:=’, ‘::=’, or ‘:::=’`.

Whitespace around the variable name and immediately after the ‘=’ is ignored. 

对于代码 `objects = main.o foo.o bar.o utils.o`, 则具体的值则是 `main.o foo.o bar.o utils.o`



