# 1. make introduction

GNU Make

## 1.1. 命令

* `-j [N], --jobs[=N]`
    * Allow N jobs at once; infineite jobs with no arg.
    * 允许并行编译
    * 不指定数字的时候自动探测最大的可用CPU
* `-f, --file` 
    * 当 makefile 文件不是默认的时候, 用该参数来指定要执行的 makefile 文件

# 2. makefile


无论是C、C++、还是pas，首先要把源文件编译成中间代码文件，在Windows下也就是 .obj 文件，UNIX下是 .o 文件，即 Object File，这个动作叫做编译（compile）。然后再把大量的Object File合成执行文件，这个动作叫作链接（link）

一般来说，每个源文件都应该对应于一个中间目标文件（O文件或是OBJ文件）。 

给中间目标文件打个包，在Windows下这种包叫“库文件”（Library File)，也就是 .lib 文件，在UNIX下，是Archive File，也就是 .a 文件
## 2.1. GCC

```
-c           编译和汇编，但不要链接。
-o <file>    将输出放入<文件> 这里的输出代表任何形式的操作输出,不论是单编译还是编译且链接都是通过 -o 来指定输出文件
'无参数'      表示恢复为基于文件扩展名猜测语言的默认行为。
```

通过gcc 不加参数可以一步直接编译生成可执行文件
`gcc main.c`  
可以通过-o选项更改生成文件的名字
`gcc main.c -o main.exe`


如果不用 makefile，则需要按照下面的方式编译上述代码：
```
g++ -c function1.cpp
g++ -c function2.cpp
g++ -c main.cpp
g++ -o hello main.o function1.o function2.o
```

## 2.2. Makefile 入门

makefile文件最好直接命名为`makefile`或`Makefile`
在该目录下直接输入命令 `make` 来使用  

当然，你可以使用别的文件名来书写Makefile，比如：“Make.Linux”，“Make.Solaris”，“Make.AIX”等，如果要指定特定的Makefile，你可以使用make的 -f 和 --file 参数，如： make -f Make.Linux 或 make --file Make.AIX
 
## 2.3. 基本概念

target也就是一个目标文件，可以是Object File，也可以是执行文件。还可以是一个标签（Label），对于标签这种特性，在后续的“伪目标”章节中会有叙述。

prerequisites就是，要生成那个target所需要的文件或是目标。

command也就是make需要执行的命令。（任意的Shell命令）

Makefile的规则:`prerequisites`中如果有一个以上的文件比`target`文件要新的话，command所定义的命令就会被执行  

```
target ... : prerequisites ...
    command
    ...
    ...
```
后续的那一行`command`定义了如何生成目标文件的操作系统命令，一定要以一个 Tab 键作为开头  

## 2.4. target

```
all:
        g++ -o hello main.cpp function1.cpp function2.cpp
clean:
        rm -rf *.o hello
```

`all`,`clean`的术语为 `target` ,也可以随意指定一个名字 例如 `abc`  

命令行输入 `make` 将默认执行**第一个** `target` （即 all）下方的命令  

如要执行清理操作，则需要输入 make clean，指定执行 clean 这个 target 下方的命令  

这个 Makefile 虽然可以省去敲命令的痛苦，却无法选择性编译源码。因为我们把所有源文件都一股脑塞进了一条命令，每次都要编译整个工程，很浪费时间。


## 2.5. prerequisites
`target ... : prerequisites ...`  
目标:目标依赖的文件  

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
逻辑:
1. 命令行输入 make ，将默认执行 all 这个 target
2. 而 all 这个 target 依赖于 hello，hello 在当前目录下并不存在，于是程序开始往下读取命令..……终于找到了 `hello` 这个 `target`
3. 正待执行 `hello` 这个 target 的时候，却发现它依赖于 `main.o`，`function1.o`，`function2.o` 这三个文件，而它们在当前目录下都不存在，于是程序继续向下执行
4. 遇到 main.o target，它依赖于 main.cpp。而 main.cpp 是当前目录下存在的文件，终于可以编译了，生成 main.o 对象文件。后面两个函数以此类推，都编译好之后，再回到 hello target，连接各种二进制文件，生成 hello 文件。

第一次编译的时候，命令行会输出：
```
g++ -c main.cpp
g++ -c function1.cpp
g++ -c function2.cpp
g++ main.o function1.o function2.o -o hello
```

证明所有的源码都被编译了一遍。假如我们对 main.cpp 做一点修改，再重新 make（重新 make 前不要 make clean），则命令行只会显示：
```
g++ -c main.cpp
g++ main.o function1.o function2.o -o hello
```

这样，我们就发挥出 Makefile 选择性编译的功能了  

对于target例如:  
`clean`:没有被第一个目标文件直接或间接关联，那么它后面所定义的命令将不会**被自动执行**  
我们可以显式执行make。即命令—— `make clean` ，以此来清除所有的目标文件，以便重编译  

# 3. makefile 变量

## 3.1. 变量
我们希望将需要反复输入的命令整合成变量，用到它们时直接用对应的变量替代，这样如果将来需要修改这些命令，则在定义它的位置改一行代码即可  

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
-Wall 表示显示编译过程中遇到的所有 `warning`  
引用变量名时需要用 $() 将其括起来，表示这是一个变量名  


在我们的makefile中以 $(objects) 的方式来使用这个变量   
```makefile
objects = main.o kbd.o command.o display.o \
    insert.o search.o files.o utils.o
edit : $(objects)
    cc -o edit $(objects)
clean :
    rm edit $(objects)
```
如果有新的 .o 文件加入，我们只需简单地修改一下 objects 变量就可以了


## 3.2. 特殊的简化符号

`$@` `$<` `$^`  
对于存在的一个段 `all: library.cpp main.cpp`  
* $@ 指代 该本段的`target`  即 `all`
* $< 指代 `library.cpp`   即第一个 dependency
* $^ 指代 library.cpp 和 main.cpp  即**所有的** dependencies

我们的 dependencies 中的内容，往往和 g++ 命令中的内容重复,例如：
```makefile
hello: main.o function1.o function2.o
        $(CC) $(LFLAGS) main.o function1.o function2.o -o hello

# 使用特殊符号简化后
hello: main.o function1.o function2.o
        $(CC) $(LFLAGS) $^ -o $@        
```

## 3.3. 自动检测目录
自动检测目录下所有的 cpp 文件呢？此外 main.cpp 和 main.o 只差一个后缀，能不能自动生成对象文件的名字，将其设置为源文件名字后缀换成 .o 的形式

`wildcard` 用于获取符合特定规则的文件名  
```makefile
SOURCE_DIR = . # 如果是当前目录，也可以不指定
SOURCE_FILE = $(wildcard $(SOURCE_DIR)/*.cpp)
#  $(wildcard *.cpp))
target:
    # 输出的为当前目录下所有的 .cpp 文件
    @echo $(SOURCE_FILE)
```
其中 @echo 前加 @是为了避免命令回显，上文中 make clean 调用了 `rm -rf` 会在 terminal 中输出这行命令，如果在 `rm` 前加了 `@` 则不会输出了  

`patsubst`用它可以方便地将 .cpp 文件的后缀换成 .o  
它的基本语法是：`$(patsubst 原模式，目标模式，文件列表) ` 
```makefile
SOURCES = main.cpp function1.cpp function2.cpp
# 目标文件都是 .o 结尾的，那么就将其表示为 %.o
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

## 3.5. 特殊target

对于`clean`, 因为没有依赖文件,所以不会被自动执行,但有更稳妥的书写格式  

* .PHONY 是一个预定义变量
  * `.PHONY : clean` 表示 clean 是一个“伪目标”

```makefile
clean:
    rm edit $(objects)

.PHONY : clean
clean :
    -rm edit $(object
```
而在 rm 命令前面加了一个小减号的意思就是，也许某些文件出现问题，但不要管，继续做后面的事  

# 4. 高级操作

## 4.1. 多makefile处理

在Makefile使用 include 关键字可以把别的Makefile包含进来，这很像C语言的 `#include`  

`include filename`
filename 可以是当前操作系统Shell的文件模式（可以包含路径和通配符）

多个filename用一个或多个空格隔开,例  
```makefile
# 想要包含的makefile文件有
# a.mk 、 b.mk 、 c.mk  foo.make 
# 一个变量 $(bar) ，其包含了 e.mk 和 f.mk

include foo.make *.mk $(bar)
#等价于
include foo.make a.mk b.mk c.mk e.mk f.mk
```

如果文件都没有指定绝对路径或是相对路径的话，make会在当前目录下首先寻找，如果当前目录下没有找到，那么，make还会在下面的几个目录下找：
1. 如果make执行时，有 -I 或 --include-dir 参数，那么make就会在这个参数所指定的目录下去寻找。
2. 如果目录 <prefix>/include （一般是： `/usr/local/bin` 或 `/usr/include` ）存在的话，make也会去找。

* 如果有文件没有找到的话，make会生成一条警告信息，但不会马上出现致命错误。它会继续载入其它的文件
* 一旦完成makefile的读取，make会再重试这些没有找到，或是不能读取的文件，如果还是不行，make才会出现**一条致命信息**。
* 如果你想让make不理那些无法读取的文件，而继续执行，你可以在include前加一个减号“-”。如：`-include <filename>`



