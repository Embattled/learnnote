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



## 3.2. 特殊符号

可以极大的简化 makefile 的书写, 同时极大的提高 makefile 的阅读门槛



## 3.3. 自动检测目录
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

## 3.4. 自动推导(简化makefile代码)

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

# 4. 高级操作

## 4.1. 多makefile处理

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



# 5. Writing Rules


## 5.1. Recipe syntax

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


## 5.2. Recipe Echoing

规则回声: 对于每一条 recipe, make 默认会在执行前把要执行的具体 shell 命令打印到 shell 输出里, 这个过程被叫做 echoing  

有时候这个特性会导致 echoing 占据了绝大部分的输出导致有用的信息无法被查看, 通过在 recipe 的起始使用 `@` , 该行 recipe 的 echoing 就会被关闭  

最基本的用途是在调用 `echo` 的时候使用 `@`, 否则相关内容就会被输出两遍   

与之相关的还有两个 make CLI option:
* `-n --just-print` : 不执行所有命令, 仅仅只是打印, 这对于差错非常有帮助, 该情况下即使有 `@` 的 recipe 也会被打印
* `-s --silent`     : 屏蔽左右 echoing, 相当于所有 recipe 都加上了 `@`, 但是是 CLI 进行控制的
* 一个 build-in target `.SILENT` 也有和 `@` 相同的效果
  

## 5.3. Types of Prerequisites 
<!-- 完 -->
Prequisites 的种类 : normal prerequisites, order-only prerequisites

这里主要特殊的地方就是 order-only prerequisites 不会影响 target 的 rebuild  
* 即, 将某一个 reprequisites 作为 Order-only 的时候, 该文件的时间戳更新不会影响 target
* 在官方文档中, 作为 obj 文件夹的创建 recipe 的时候似乎很有用处

```makefile
OBJDIR := objdir
OBJS := $(addprefix $(OBJDIR)/,foo.o bar.o baz.o)

$(OBJDIR)/%.o : %.c
        $(COMPILE.c) $(OUTPUT_OPTION) $<

all: $(OBJS)

# 每个OBJS 都 order-only 依赖于 mkdir
$(OBJS): | $(OBJDIR)

$(OBJDIR):
        mkdir $(OBJDIR)
```


## 5.4. Using Wildcard Characters in File Names
<!-- 完 -->
在文件名中使用 Wildcard 来进行文件名通配符匹配  
* 匹配得到的文件列表会被排序
* wildcard 可以后接多个 匹配项 `*.c *.h` 多个匹配项得到的文件名会按照顺序拼接, 最终并不会得到一个全局排序的列表
* `~` 可以被用在通配符表达式里, 用于表示 home 文件夹, 在 Windows 下会参照 HOME 环境变量 
* 在 recipes 里, 通配符会直接被 shell 所自动执行
* 在 其他地方下, 需要手动调用 `wildcard` 函数来让 make 执行通配符匹配

```makefile
# 在 .Phone clean 下定义清除 recipe
clean:
        rm -f *.o

# 利用 prerequisite 来查看所有更改后的 .c 源文件名称  
print: *.c
        lpr -p $?
        touch print

# 在其他地方 通配符不会被自动扩展  
objects = *.o   # objects 的值就是 *.o
foo : $(objects) # 在这种情况下, objects 的值被替换后因为是在 recipe 中, 所以最终仍然会被 shell 所执行替换, 但是这很容易导致 bug 是个陷阱

objects := $(wildcard *.o) # 需要手动调用 wildcard 函数才能正常通配 objects 

# 结合 patsubst 来自动生成所有 .o 路径
objects := $(patsubst %.c,%.o,$(wildcard *.c))

foo : $(objects)
        cc -o foo $(objects)
```

## 5.5. Searching Directories for Prerequisites

对于大型项目, 通常会将 源代码 和 二进制 文件放于不同的目录中, make 提供了目录搜索功能:
* 通过自动搜索多个目录来查找 prerequisites  
* 当目录之间重新分配文件时, 不需要更改单独的规则, 只需要更改搜索路径  
* 所有的搜索都发生在 target or prerequisites 不在当前目录的情况下

### 5.5.1. VPATH: Search Path for All Prerequisites

`VPATH` 是一个 make variable, 用于指定 list of directories that make should search.  
* 多数情况下 : `VPATH` 用于指定那些目录 包含了 `prerequisite files` that are not in the current directory
* 事实上从 make 的功能上 : make uses `VPATH` as a search list for `both` prerequisites and targets of rules

如果某一个 file 被指定为 target 或者 prerequisite 并且不在当前目录下, make 则会在 VPATH 的目录下寻找它们. 对于被寻找到的 file, 可以在 makefile 中直接使用文件名( 不需要前加相对目录), 就好像这些文件在当前目录下一样  

VPATH 中目录的定义顺序等同于 make 的搜索顺序, 目录间的分隔符号可以使用冒号(windows 下使用分号) 和空格 

```makefile
# 对于没有定义 VPATH 的情况下
# foo.o : src/foo.c 需要指定相对目录
# 定义了 VPATH 则可以直接指定文件夹  
VPATH = src:../headers
foo.o : foo.c
```

### 5.5.2. The vpath Directive

`vpath` (小写), 是类似与 `VPATH`, 但是更加具有选择性的一种目录指定方法, 可以对不同的文件类型指定不同的搜索目录  

具体的用法有三种  
* `vpath pattern directories`  对于 指定的 pattern 定义搜索目录
* `vpath pattern`              取消对一个 pattern 的搜索目录的定义
* `vpath`                      取消所有 vpath 定义  

vpath 使用的关键在于 pattern 的定义, 对于一个 pattern
* a string containing a `%` character. `%` 会匹配 0个或以上的字符. pattern 可以不带 % , 但是那样的话就没啥用了  
* `%` 本身可以通过 `\` 反斜杠来 转义
* pattern 不需要唯一, 即同样的 pattern 也可以被多次定义, 这些 directive 之间彼此独立. 某一个 prerequisite 也可以被多个 pattern 所匹配, 最终的搜索会按照 rule 的定义顺序, 目录的从左到右来搜索.  
* 


## 5.6. Phony Target 

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


## 5.7. Special Built-in Target Names

一些特殊的 name 作为 Target 的时候会有特殊的意思

像是 `clean`, 因为没有依赖文件,所以不会被自动执行,但有更稳妥的书写格式  


* .PHONY : prerequisites of the special target .PHONY are considered to be phony targets


```makefile
clean:
    rm edit $(objects)


```
而在 rm 命令前面加了一个小减号的意思就是, 也许某些文件出现问题, 但不要管, 继续做后面的事  

# 6. How to Use Variables

<!-- (头部完) -->
makefile 的 variable 是一些被定义为 name 的 string or text, 这些 varable 的值可以替换在任何部分 (target, prerequisites, recipes, etc.)

在一些其他版本的 make 里, variables 也被叫做 macros

variables 的命令规则:
* 不能有 `: # =` 符号
* 以点 `.` 和 大写字母开头的 name 有可能会有特殊的意思
* variable 的命名区分大小写
* 传统上, 为内部使用的变量赋予小写字母命名, 而对于 控制隐式规则的参数 保留大写字母

单个特殊符号的变量有特殊用途 (automatic variables)

## 6.1. Basics of Variable References
<!-- 完 -->
我们希望将需要反复输入的命令整合成变量, 用到它们时直接用对应的变量替代, 这样如果将来需要修改这些命令, 则直接修改变量的值即可

makefile 的变量使用类似于 shell
* 定义的时候使用 `=` 进行赋值
* 具体进行变量值替换的时候使用 `$(变量名)` or `${变量名}`
* 对于文件命中包含特殊符号 `$` 的时候, 使用 `$$` 来进行转义
* 对于变量的使用官方推荐加上括号, 除非在使用 automatic variable 的时候, 不加括号反而可以提高易读性


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

## 6.2. The Two Flavors of Variables
<!-- 头部完 -->
对于不同的变量赋值方法, 在 GUN make 被称为 `flavors`, 不同的 flavor 会影响这些变量值在之后的  used and expanded.

### 6.2.1. Recursively Expanded Variable Assignment - 递归扩张
<!-- 完 -->
`recursively expanded` flavor,  使用 `=` 单等号 或者 `define` 关键字 来赋值

该 flavor 是绝大多数版本的 make 都支持的一种方法, 在赋值的时候会递归的查找 值 里面的变量
```makefile
foo = $(bar)
bar = $(ugh)
ugh = Huh?

all:;echo $(foo)  # Huh?

# 无法引用自己来实现值的拓展, 会导致无线递归
CFLAGS = $(CFLAGS) -O
```

作为一种最基础的赋值方法:
* 优点: 会切实的按照变量的方法来使用, 基础
* 缺点: 
  * 无法进行变量的拓展
  * 会在每次调用的时候都对内部的变量进行拓展, 即如果是 wildcard 等函数, 不仅会导致重复的调用, 还会导致由于函数调用的时机不同导致的结果不同进而使得 makefile 的行动无法预测

### 6.2.2. Simply Expanded Variable Assignment
<!-- 完 -->
主要用于避免 `recursively expanded` flavor 缺点的 另一种主流 flavor

定义的时候使用 `:=` or `::=`, 这两种意思完全相同  

最主要的特点是: 在定义的时候进行值的变量的拓展, 所以对于函数都是只在定义的时候执行一次  
* 使得行动更加贴近于传统的编程语言
* 变量可以使用其自己的值进行重复定义, 即 内容拓展

```makefile
x := foo
y := $(x) bar  # y := foo bar
x := later     # x := later
```

### 6.2.3. Immediately Expanded Variable Assignment

TODO

### 6.2.4. Conditional Variable Assignment
<!-- 完 -->
条件赋值 `?=` , 这里的条件指的是 左侧的变量还没有定义 (即使变量赋予的是空值也仍然属于已定义的变量)  

```makefile
FOO ?= bar

# 相当于
ifeq ($(origin FOO), undefined)
  FOO = bar
endif
```

## 6.3. Advanced Features for Reference to Variables

使用变量的更高级的技巧  

### 6.3.1. Substitution References

替换引用:  `$(var:a=b)`  or `${var:a=b}`

是一种很精妙的赋值方法: 具体的效果为  提取 var 的当前值, 将 var 的值里面的每一个 word 最后的 `a` 替换为 `b`, 并最终返回替换后的值  
* replace every `a` at the end of a word with `b` in that value, and substitute the resulting string. 
* 文档中的 `at the end of a word`  指的是 `a` 必须出现在:
  * 后接空格的地方, 即 end of a word
  * end of the value
  * 主要是为了防止歧义
* 这种方式的赋值主要是函数 `patsubst` 的简短写法
  * `$(var:a=b)` is equivalent to `$(patsubst %a,%b,var)`
  * 注意这里的相当于的 patsubst 里的 a,b 都前接了一个 `%`
* 同时还有另一种完全使用 `patsubst` 特性的写法, 其要求 `a` 必须包含 `%` 字符 
  * `bar := $(foo:%.o=%.c)`

```makefile
foo := a.o b.o l.a c.o
bar := $(foo:.o=.c)  # bar =  a.c b.c l.a c.c

bar := $(foo:%.o=%.c) # bar = a.c b.c l.a c.c
```


## 6.4. Setting Variables 设置变量

To set a variable from the makefile, write a line starting with the variable name followed by one of the assignment operators` ‘=’, ‘:=’, ‘::=’, or ‘:::=’`.  
Whitespace around the variable name and immediately after the ‘=’ is ignored.   
对于代码 `objects = main.o foo.o bar.o utils.o`, 则具体的值则是 `main.o foo.o bar.o utils.o`


# 7. Functions for Transforming Text

make 本身提供了可以用于 makefile 里的许多函数, 这些函数大部分都是用来处理 字符串的, 即用于处理各种文件名或者路径  

同时, 用户也可以自定义各种函数  


## 7.1. Function Call Syntax

函数的调用和 变量的使用类似, 但是因为函数有参数, 所以实际的写法 为`函数名 参数` 被圆括号或者大括号括起   

```makefile
$(function arguments)
${function arguments}
```

* 函数名和参数之间通过 空格或者 tab 分隔, 而参数之间通过 逗号 `,` 分隔, 
* 某个参数包括多个文件名的时候, 文件名之间通过空格分隔
* 函数调用可以嵌套使用, 但是圆括号和大括号不能混用
* 特殊字符不能被 转义, 但是可以通过定义一个变量的形式来隐藏
  * 逗号
  * 首个字母为空格
  * 不匹配的括号


## 7.2. Functions for String Substitution and Analysis

主要用于字符串处理的一些函数  

索引:
* filter                : 过滤掉不满足 pattern 的, removing any words that do not match




### 字符串替换

* `$(subst from,to,text)`
* `$(patsubst pattern,replacement,text)`
  

### 字符串调整

* `$(strip string)`
* `$(findstring find,in)`
* `$(filter pattern…,text)`

### 值列表调整

* `$(sort list)`
* `$(word n,text)`
* `$(firstword names…)`
* `$(lastword names…)`


## 7.3. Functions for File Names
<!-- 完 -->

专门用于针对文件的路径进行处理的函数, 包括分离文件名的各个部分等  

输入的参数被视为通过空格分割的一系列文件名, 同时输出也是同理  


提取函数
* `$(dir names…)`       : 提取文件夹路径, 即最后一个 `/` 之前的部分, 如果没有 `/`, 则文件夹路径视作 `./`
* `$(notdir names…)`    : 提取非文件夹路径的部分, 即最后一个 `/` 之后的部分, 如果没有 `/` 则不做任何更改
* `$(suffix names…)`    : 提取后缀, 如果文件没有后缀则返回空, 即返回的数值个数可能会变少
* `$(basename names…)`  : 提取非后缀的部分, 同理如果没有后缀则不更改

执行结果
* `$(dir src/foo.c hacks)`   `src/ ./` 
* `$(notdir src/foo.c hacks)`  `foo.c hacks`
* `$(suffix src/foo.c src-1.0/bar.c hacks)`  `.c .c`
* `$(basename src/foo.c src-1.0/bar hacks)`  `src/foo src-1.0/bar hacks`

路径修改函数
* `$(addsuffix suffix,names…)`          : 追加后缀
* `$(addprefix prefix,names…)`          : 追加前缀
* `$(join list1,list2)`                 : 两个 list 的元素依次结合, 如果元素个数不匹配, 尾部剩余的部分则会原样拷贝  

示例
* `$(addsuffix .c,foo bar)`     `foo.c bar.c`
* `$(addprefix src/,foo bar)`   `src/foo src/bar`
* `$(join a b,.c .o)`           `a.c b.o`


路径转换:
* `$(wildcard pattern)`                  : 通配符手动调用函数
* `$(realpath names…)`                   : 路径转换, 包括转换链接, 消除 `../`, 消除重复的 `/`, 验证路径是否存在, 如果转换失败则返回空字符串
* `$(abspath names…)`                    : 有些类似于 realpath, 但不进行验证存在, 同时不进行链接转换  

# 8. Using Implicit Rules - 10


You can define your own implicit rules by writing `pattern rules`.    
`Suffix rules` are a more limited way to define implicit rules.  
`Pattern rules` are more general and clearer, but suffix rules are retained for compatibility. 

## 8.1. Defining and Redefining Pattern Rules

通过定义 Pattern rule 可以实现对 implicit rule 进行定义或重定义  

Pattern rule: 从形式上和 ordinary rule 没有区别, 但是包含了 `%` 符号  

Pattern rule 的 `%` 扩展发生在 变量替换和 函数执行 之后

### 8.1.1. Introduction to Pattern Rules

一个 rule 被称为 pattern rule 是因为它的 target 里包含一个 `%`, 可以匹配任何 nonempty substring, 被 match 的部分被叫做 `stem`  
A pattern rule contains the character ‘%’ (exactly one of them) in the target


`%` 的匹配替换是以 target 为主的, 即 `%` 也可以出现在 prerequisite 里, 但是 stem 的值是依据 target 的


### 8.1.2. Automatic Variables

让 makefile 变成天书的罪魁祸首, `automatic variables` 根据每一项 rule 来重新计算该 variable 的值  

automatic variable 仅仅只在 recipe 里有效, 让 prerequisite list 里也能使用 `automatic variables` 的特性称为 `Secondary Expansion` 在之前的章节里介绍


对于存在的一个段 `all: library.cpp main.cpp`  
* `$@` : 本段的`target`  即 `all`, 对于具有多个目标的 pattern rule, `$@` 也是根据 rule 来确定要运行对应 recipe 时候的 target
* `$<` : `library.cpp`   即第一个 prerequisite, 即使 prerequisite 是根据 implicit 来自动添加的, 也是指的是第一个 
* `$^` : library.cpp 和 main.cpp  即**所有的** dependencies

我们的 dependencies 中的内容, 往往和 g++ 命令中的内容重复,例如: 
```makefile
hello: main.o function1.o function2.o
        $(CC) $(LFLAGS) main.o function1.o function2.o -o hello

# 使用特殊符号简化后
hello: main.o function1.o function2.o
        $(CC) $(LFLAGS) $^ -o $@        
```