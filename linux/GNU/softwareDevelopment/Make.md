- [1. make](#1-make)
- [2. makefile](#2-makefile)
  - [2.1. 基础 makefile](#21-基础-makefile)
- [3. Writing Makefiles](#3-writing-makefiles)
  - [3.1. Including Other Makefiles - 导入其他 makefile](#31-including-other-makefiles---导入其他-makefile)
- [4. Writing Rules - 编写规则](#4-writing-rules---编写规则)
  - [4.1. Rule Example](#41-rule-example)
  - [4.2. Rule Syntax](#42-rule-syntax)
  - [4.3. Types of Prerequisites](#43-types-of-prerequisites)
  - [4.4. Using Wildcard Characters in File Names](#44-using-wildcard-characters-in-file-names)
  - [4.5. Searching Directories for Prerequisites](#45-searching-directories-for-prerequisites)
    - [4.5.1. VPATH: Search Path for All Prerequisites](#451-vpath-search-path-for-all-prerequisites)
    - [4.5.2. The vpath Directive](#452-the-vpath-directive)
  - [4.6. Phony Target](#46-phony-target)
    - [4.6.1. Phony with recursive make](#461-phony-with-recursive-make)
    - [4.6.2. Phony have prerequisites](#462-phony-have-prerequisites)
  - [4.7. Rules without Recipes or Prerequisites](#47-rules-without-recipes-or-prerequisites)
  - [4.8. Special Built-in Target Names](#48-special-built-in-target-names)
  - [4.9. Multiple Targets in a Rule](#49-multiple-targets-in-a-rule)
  - [4.10. Multiple Rules for One Target](#410-multiple-rules-for-one-target)
  - [4.11. Static Pattern Rules - 静态 Pattern Rules](#411-static-pattern-rules---静态-pattern-rules)
    - [4.11.1. Syntax of Static Pattern Rules - 静态 Pattern Rules 语法](#4111-syntax-of-static-pattern-rules---静态-pattern-rules-语法)
    - [4.11.2. Static Pattern Rules versus Implicit Rules - 与 Implicit Rules 的比较](#4112-static-pattern-rules-versus-implicit-rules---与-implicit-rules-的比较)
- [5. Writing Recipes in Rules - 在 recipes 中编写 规则](#5-writing-recipes-in-rules---在-recipes-中编写-规则)
  - [5.1. Recipe syntax](#51-recipe-syntax)
  - [5.2. Recipe Echoing](#52-recipe-echoing)
- [6. How to Use Variables - makefile 的变量](#6-how-to-use-variables---makefile-的变量)
  - [6.1. Basics of Variable References](#61-basics-of-variable-references)
  - [6.2. The Two Flavors of Variables](#62-the-two-flavors-of-variables)
    - [6.2.1. Recursively Expanded Variable Assignment - 递归扩张](#621-recursively-expanded-variable-assignment---递归扩张)
    - [6.2.2. Simply Expanded Variable Assignment](#622-simply-expanded-variable-assignment)
    - [6.2.3. Immediately Expanded Variable Assignment](#623-immediately-expanded-variable-assignment)
    - [6.2.4. Conditional Variable Assignment](#624-conditional-variable-assignment)
  - [6.3. Advanced Features for Reference to Variables](#63-advanced-features-for-reference-to-variables)
    - [6.3.1. Substitution References](#631-substitution-references)
  - [6.4. Setting Variables 设置变量](#64-setting-variables-设置变量)
  - [6.5. Defining Multi-Line Variables - 多行变量与 define 关键字](#65-defining-multi-line-variables---多行变量与-define-关键字)
  - [6.6. Undefining Variables - 取消一个变量的定义](#66-undefining-variables---取消一个变量的定义)
  - [6.7. Variables from the Environment - 环境变量](#67-variables-from-the-environment---环境变量)
- [7. Conditional Parts of Makefiles - Makefile 的分支执行](#7-conditional-parts-of-makefiles---makefile-的分支执行)
  - [7.1. Example of a Conditional - 快速实例](#71-example-of-a-conditional---快速实例)
  - [7.2. Syntax of Conditionals - 完整的 makefile 条件语法](#72-syntax-of-conditionals---完整的-makefile-条件语法)
  - [7.3. Conditionals that Test Flags -](#73-conditionals-that-test-flags--)
- [8. Functions for Transforming Text](#8-functions-for-transforming-text)
  - [8.1. Function Call Syntax](#81-function-call-syntax)
  - [8.2. Functions for String Substitution and Analysis](#82-functions-for-string-substitution-and-analysis)
    - [8.2.1. 字符串替换](#821-字符串替换)
    - [8.2.2. 字符串调整](#822-字符串调整)
    - [8.2.3. 值列表调整](#823-值列表调整)
  - [8.3. Functions for File Names](#83-functions-for-file-names)
  - [8.4. Functions That Control Make - 控制 make 执行的函数](#84-functions-that-control-make---控制-make-执行的函数)
- [9. How to Run make - make CLI 文档](#9-how-to-run-make---make-cli-文档)
  - [9.1. Arguments to Specify the Makefile](#91-arguments-to-specify-the-makefile)
  - [9.2. Arguments to Specify the Goals](#92-arguments-to-specify-the-goals)
  - [9.3. Overriding Variables](#93-overriding-variables)
- [10. Using Implicit Rules](#10-using-implicit-rules)
  - [10.1. Defining and Redefining Pattern Rules](#101-defining-and-redefining-pattern-rules)
    - [10.1.1. Introduction to Pattern Rules](#1011-introduction-to-pattern-rules)
    - [10.1.2. Automatic Variables](#1012-automatic-variables)
- [11. Using make to Update Archive Files](#11-using-make-to-update-archive-files)

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


官方文档
https://www.gnu.org/software/make/manual/make.html

# 2. makefile


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



## 2.1. 基础 makefile 


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






# 3. Writing Makefiles

## 3.1. Including Other Makefiles - 导入其他 makefile


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


# 4. Writing Rules - 编写规则

<!-- 头部完 -->

`rule` : 用于在 makefile 中指定 何时/如何 更新指定的文件( 即target ), target 一般在一条 rule 中只会有一个  , 是 makefile 中模块的单位名称  

rule 的编写顺序不重要.  除了所谓的 `default goal` : make 所默认要执行的首个 target. 除非显式指定, 否则为 makefile 文件中出现的第一个 rule 中的 target.  
   The default goal is the first target of the first rule in the first makefile   

关于 default goal 的默认选定, 还有两个例外
* 以 period `.` 开头的 rule 即使作为 首个 rule 也不会被指定为 default goal ( 因为这是 phony target)
  * 除非 `.` 是被转义的 `/.`
* 如果 target 是一个 pattern rule. 则也不会被作为 default goal ( pattern 会指定多个文件为 target )

因此约定俗成的, makefile 的第一个规则是编译整个程序 或者最终目标, 且一般命名为 `all`  

## 4.1. Rule Example
<!-- 完 -->
rule 是 target, prerequisites, recipe 所构成的, 更新文件的方法的集合(规则)

```makefile
foo.o : foo.c defs.h       # module for twiddling the frobs
        cc -c -g foo.c
```

## 4.2. Rule Syntax
<!-- 完 -->
rule 由 3 部分构成  , rule 的目标是 : 用于确定 target 何时/是否过时, 并且如何更新 target

```makefile
targets : prerequisites
        recipe
        …

targets : prerequisites ; recipe
        recipe
        …
```

target : file names, separated by spaces
* 可以在 file name 里使用通配符
* 可以使用 archive file 的格式,  e.g. `a(m)` 代表在名为 a 的 archive 的文件中查阅名为 m 的成员
* 通常情况下, target 只会由一个目标, 但是也允许存在多个目标, 参考 Multiple Targets in a Rule

recipe : 以制表符开头的 可执行的 command 
* 除了制表符外, 还可以参阅特殊变量 `.RECIPEPREFIX` 确认其他可以使用的 前缀
* 首个 recipe 除了出现在 prerequisites 之后的行上, 还可以接在 prerequisites 正后方, 以分号作为分隔符
* 详细的 recipe 文档在下一章学习  

prerequisites : 由空格分隔的文件名, 同样允许 通配符和 archive member
* 决定 target 的过失标准
* 如果 target 的文件不存在, 或者时间戳早于任何 prerequisites, 则说明目标已过时  

关于行长: make 不限制一行的长度, 可以为了增加易读性而手动通过反斜杠来分隔行 

关于美元符号 `$` 的转义
* 如果希望在 target 或者 prerequisites 的文件名中使用 美元符号, 则转义非常的麻烦
* 一般的转义需要书写两个美元符号 `$$`
* 如何启用了所谓为的 `Secondary Expansion` 则需要书写 4 个美元符号 `$$$$`

## 4.3. Types of Prerequisites 
<!-- 完 -->
Prequisites 的种类 : normal prerequisites, order-only prerequisites

```makefile
targets : normal-prerequisites | order-only-prerequisites
```
normal prerequisites: 
* 规定了配方的调用顺序: target 的所有 prerequisites 都会在该 target 的 recipe 调用之前被完成
* 如果任何 prerequisites 比 target 新, 则 target 需要被更新


这里主要特殊的地方就是 order-only prerequisites 不会影响 target 的 rebuild  
* 即, 将某一个 prerequisites 作为 Order-only 的时候, 该文件的时间戳更新不会影响 target
* 即使该 prerequisites 是一个 .phony , 也不会影响 target 的更新
* 在官方文档中, 用作文件夹的创建 recipe 的时候似乎很有用处
  * 文件夹的时间戳会随着文件夹中  文件的 增删改 而随之变化
  * 因此 文件夹的时间戳不适合作为 target 的更新依据


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


## 4.4. Using Wildcard Characters in File Names

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

## 4.5. Searching Directories for Prerequisites

对于大型项目, 通常会将 源代码 和 二进制 文件放于不同的目录中, make 提供了目录搜索功能:
* 通过自动搜索多个目录来查找 prerequisites  
* 当目录之间重新分配文件时, 不需要更改单独的规则, 只需要更改搜索路径  
* 所有的搜索都发生在 target or prerequisites 不在当前目录的情况下

### 4.5.1. VPATH: Search Path for All Prerequisites

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

### 4.5.2. The vpath Directive

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


## 4.6. Phony Target 
<!-- 完 -->

Phony Target 是一种特殊的目标, 他并不是一个真实的生成目标文件, 更像是一个 recipe 的集合的命名, 主要用于 make 执行具体的 request 的时候  
* to avoid a conflict with a file of the same name, 如果不定义成 phony
* and to improve performance

```makefile
clean:
        rm *.o temp

.PHONY : clean
clean :
    -rm edit $(object)
```

对于常用的 clean rule: 如果不定义成 phony, 那么随着同名为 clean 的有无, 会有两种极端情况  
* 由于其没有 prerequisites, 且他的 recipe 不会生成名为 clean 的文件, 所以 make 会在每次调用 `make clean` 的时候都执行 `rm` 命令  
* 然而若是当前目录中存在了名为 `clean` 的任意文件, 则由于没有 prerequisites, 导致 make 认为 clean 已经是最新的, 从而导致相应的 rm 命令被跳过
* 解决办法就是, 将 clean rule 实现声明为 Phony Target
* the phony target recipe will be executed only when the phony target is a specified goal (see Arguments to Specify the Goals). 

声明为 phony 后, 执行 `make <target>` 则会无视 clean 文件的存在与否 执行对应的 recipe

注意: 声明为 phony 后:
* 该 rule 的 prerequisites 将始终被解释为 `literal target name`. 即 永远不会被解释为 pattern (即使包含了 `%` 字符也不会)
* 该 rule 不会被 `implicit rule search` 搜索到, 因此定义为 phony 之后就不用再担心 名为 target 的文件存在的情况  
* 反过来, 声明为 phony 的 target 也不应该成为别的 `非 phony rule` 的 prerequisites
  * 因为不考虑是否存在对应文件以及时间戳, 因此会导致 phony 总是执行被执行
  * 需要慎重考虑该用法
* 如果要建立一个 无视 prerequisites 时间戳 始终重建的 pattern rule, 应该使用 `force target` 而非 `phony`
  * (see Rules without Recipes or Prerequisites). 

### 4.6.1. Phony with recursive make
<!-- 完 -->

除了 clean 等 target 常被定义为 phony 以外, 和 make 的递归调用相关的 rule 也适合定义为 phony

```makefile
# 定义一些 子目录, 需要递归的执行对应的子 makefile
SUBDIRS = foo bar baz

# 传统方法, 通过将循环写在 recipe 中实现
subdirs:
        for dir in $(SUBDIRS); do \
          $(MAKE) -C $$dir; \
        done
# 缺点有
#   1. 无法自由的判断子 makefile 的运行情况对 主 make 的影响, 例如通过 shell 命令来获取子 make 的返回值, 但这又会导致 `-k` 命令的失效
#   2. 无法利用 make 的并行化功能, recipe 只有一条规则, 该 for 只会在单个线程上运行

# phony 方法, 通过将整个 子目录变量定义为 phony, 相当于定义了每一个 子目录的 phony target
.PHONY: subdirs $(SUBDIRS)
# phony 成为了另一个 phony 的 prerequisites, 这是合理的
subdirs: $(SUBDIRS)
# 可以使用 auto variable
$(SUBDIRS):
        $(MAKE) -C $@
# 定义了 foo 和 baz 的先后关系
foo: baz
```

### 4.6.2. Phony have prerequisites

一个最常用的 phony 就是 `all`, 将其写作首个 rule 的 target, 并定义为 phony, 再将构建所有子程序的 target 作为 prerequisites 传入该 rule 即可  
如上节所说, 如果 phony 的 prerequisites 也是 phony , 则就是很普通的作为 subroutine 被执行  

```makefile
# phony 的 all 需要其他三个子程序, 这样 make 的默认 target 就是 all
all : prog1 prog2 prog3
.PHONY : all

prog1 : prog1.o utils.o
        cc -o prog1 prog1.o utils.o

prog2 : prog2.o
        cc -o prog2 prog2.o

prog3 : prog3.o sort.o utils.o
        cc -o prog3 prog3.o sort.o utils.o
```

## 4.7. Rules without Recipes or Prerequisites

规则上: 
* 如果一个 target 不存在 prerequisites 或者 recipes, 同时不存在名为 `<target>` 的文件, 那么 make 会认为该 target 在 make 运行的时候已经被更新
* 从而导致 所有依赖于该 target 的其他 target 也被更新
* 这听起来很像 `PHONY`, 事实上, 该规则的利用就是某种程度上对于不支持 PHONY 的 make 版本的兼容
* 约定俗成的, 该 target 总是被命名为 `FORCE`


```makefile
# 定义 force rule
clean: FORCE
        rm $(objects)
# 定义 FORCE
FORCE:
```

## 4.8. Special Built-in Target Names

一些特殊的 name 作为 Target 的时候会有特殊的意思, 例如作为这些 target 的 prerequisites 则会被设定为 特殊的 target, 例如 PHONY



* `.PHONY`
* `.SUFFIXES`
* `.DEFAULT`
* `.PRECIOUS`
* `.INTERMEDIATE`
* `.NOTINTERMEDIATE`
* `.SECONDARY`
* `.SECONDEXPANSION`
* `.DELETE_ON_ERROR`
* `.IGNORE`
* `.LOW_RESOLUTION_TIME`
* `.SILENT`
* `.EXPORT_ALL_VARIABLES`
* `.NOTPARALLEL`
* `.ONESHELL`
* `.POSIX`

## 4.9. Multiple Targets in a Rule

## 4.10. Multiple Rules for One Target


## 4.11. Static Pattern Rules - 静态 Pattern Rules

指定多个 target, 同时根据 target 的名称构建每个目标自己的 prerequisites
* 对于普通的 multiple targets rule, 其 prerequisites 只能指定成相同的
* static pattern rules 对于每个 target 都会生成根据对应 pattern 生成的 prerequisites, 这更加自由

### 4.11.1. Syntax of Static Pattern Rules - 静态 Pattern Rules 语法


```makefile
targets …: target-pattern: prereq-patterns …
        recipe
        …
```
这里的targets代表一组target  例如  
`OBJS = $(patsubst %.cpp, %.o, $(wildcard *.cpp))`  


* target-pattern 代表target的特征
* prereq-patterns 代表原本的依赖文件的特征

最终通过一行可以执行全部的文件编译 
```makefile
$(OBJS):%.o:%.cpp
        $(CC) $(CFLAGS) $< -o $@

```

### 4.11.2. Static Pattern Rules versus Implicit Rules - 与 Implicit Rules 的比较



# 5. Writing Recipes in Rules - 在 recipes 中编写 规则

首先定义 recipe of a rule : one or more shell command lines to be executed, 一次一个按照出现的顺序执行.   
通常情况下, 所谓 的 recipe of a rule 的目的是让 target of the rule 进行更新  

要注意, 默认情况下 makefile 中的 recipes 是由 `/bin/sh` 解释的, 除非特殊指定, 因此在编写的时候要注意命令的兼容性  


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






# 6. How to Use Variables - makefile 的变量

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

# 无法引用自己来实现值的拓展, 会导致无限递归
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

仅当 左侧的变量还处于未定义的状态, 才会将右侧的值赋予左侧  

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
* 定义一个变量的值支持 4 种赋值运算符， 

Whitespace around the variable name and immediately after the ‘=’ is ignored.    
对于代码的整洁度来说, 赋值运算符号后面的首个空格会被忽视, 然而要注意仅仅是首个会被忽视. 此外, 变量名的左右空格也会被忽视     
对于代码 `objects = main.o foo.o bar.o utils.o`, 则具体的值则是 `main.o foo.o bar.o utils.o`   


## 6.5. Defining Multi-Line Variables - 多行变量与 define 关键字
<!-- 完 -->
define 关键字是另一种完全不同的 语法 用来定义一个变量, 其特点就是支持变量内容支持 newline 字符

对于:
* canned sequences of commands  (查阅  Defining Canned Recipes )
* sections of makefile syntax to use with `eval` (查阅 The eval Function)
都是非常方便的一种方式  

语法:
* `define` 关键字后接 变量的名称和 (可选的赋值运算符), 并在 `下一行` 开始定义具体的值内容, 值的结束用一个仅包含 `endef` 的行结尾  
* 除了语法的差异以外, 与通常的变量没有其他区别.  如果变量名称中包括了函数或者引用, 则实际执行的时候会拓展这些引用
* endef 的前一行的 `换行符 newline` 不会被包括在变量值中, 因此如果需要变量结尾包括换行符则需要多插入一行空行
* (可选的赋值运算符)
  * 如果省略, make 会假定其为 `=` 递归扩展变量
  * 如果使用 `+=`, 则与其他附加操作一样, 改值将附加到前一个值并以 空格分隔
* define 的嵌套是被允许的
  * 然而, 如果以制表符开头 `recipe prefix 字符`, 则关键字会被是做 recipe 的一部分, 要注意


```makefile
# 包含一个空行 (带结尾换行符) 的变量, 需要在关键字中插入两个空行
define newline


endef

# example
define two-lines
echo foo
echo $(bar)
endef


# 从结果上, 上面的定义内容与下面是相同的
two-lines = echo foo; echo $(bar)

# 覆盖环境变量
override define

endef
```




## 6.6. Undefining Variables - 取消一个变量的定义  

<!-- 完 -->
不论一个变量是 undefine 或者值为空.  对其进行 expand 都会返回一个 空字符串, 这使得很多时候要取消一个变量的值将其置空即可足够使用   

然而对于两个函数 `flavor` 和 `origin` 来说, 这两种函数对于变量的处理结果会随着变量是否定义过而不同, 因此, 某些场景下 undefine variable 是需要的  
通过结合 `override` 关键字可以覆盖取消环境变量的定义

```makefile
foo := foo
bar = bar

undefine foo
undefine bar

$(info $(origin foo))
$(info $(flavor bar))
# 此时 上面两个函数的输出都是 "undefined"

override undefine CFLAGS
# 通过 override 关键字来强行 undefine 一个 命令行 环境变量
```

## 6.7. Variables from the Environment - 环境变量
<!-- 完 -->
make 在启动的时候会读取所有的终端环境变量, 并将其转化为对应的 make 变量.  在其后对变量进行赋值则会自然的覆盖掉从环境变量中获取的值
* 此处有一个行为改变的 option `-e`, 官方文档并不推荐使用因此不阅读

通常, 开发者会在环境变量中设置 `CFLAGS`, 因为 CFLAGS 在所有 makefile 中约定俗成的只会有一种用途. 除非某些 makefile 会在内部显式的给 CFLAGS 赋值, 则会覆盖掉环境变量中的设置  

对于 recipe 中的 命令的环境, 只有 make 本身的环境变量以及 make 自己设置的变量会作为环境变量传入 recipe 的环境中.  
某些情况下 (sub-make) 可能需要使用 export 来将 make 中的变量导入到外部 环境中,  See `Communicating Variables to a Sub-make`, for full details. 

甚至对于某些特殊的环境变量 , 例如 `SHELL`, make 会以特殊的方式处理, 因此在 rule 中使用某些特殊的环境变量是非常不明智的.  
 see `Choosing the Shell`. 

# 7. Conditional Parts of Makefiles - Makefile 的分支执行

<!-- 头部完 -->

makefile 中的条件分歧, 根据变量值来使得 makefile 中的一部分指令被执行或者忽视
* 条件决定了 make 实际读取到的 makefile 的内容, 即可以把条件理解为 编译过程中的预处理指令
* 因此 makefile 中的条件分歧并不能实现在 make 执行过程中根据计算出来的变量产生条件分歧

比较简单的一个章节

## 7.1. Example of a Conditional - 快速实例

<!-- 完 -->
一个根据编译器是 gcc 还是其他的编译器, 来决定链接库的 makefile 示例:

```makefile
libs_for_gcc = -lgnu
normal_libs = # other libs

foo: $(objects)
ifeq ($(CC),gcc)
        $(CC) -o foo $(objects) $(libs_for_gcc)
else
        $(CC) -o foo $(objects) $(normal_libs)
endif

```

从示例中可以看出, 条件分歧使用了三个指令 `ifeq` `else` `endif` 有点类似于 shell 

`ifeq` 的用法包含两个参数, 用逗号分隔它们并用括号包围起来  `ifeq ($(CC),gcc)`
* 对于两个参数各自执行变量替换后
* 对它们进行匹配比较
* 如果为真, 则执行对应 ifeq 后面的 指令

`else` 在分歧中是可选的模块, 在 ifeq 的结果为假的时候执行该部分的内容   

`endif` 意为 条件部分的终止, 是必须的, 每个 conditional 必须以 `endif` 结尾   

## 7.2. Syntax of Conditionals - 完整的 makefile 条件语法
<!-- 完 -->
多重 if 分支和 C 语言的构成类似, 每一个分支下的代码数量都是任意的     
* 再次强调, makefile 的分支类似于 预编译指令, 对于一切 make 执行中的变量都是不生效的   

为了防止混乱
* make 不允许将条件的开始和中止分别写在不同的 文件中并 分别 include 他们
* 但是允许在 分支条件表达式里使用 include

关键字
* `ifeq         表达式`    验证表达式为真, 执行接下来的代码
* `ifneq        表达式`    验证表达式为假, 则执行接下来的代码
* `ifdef        变量名` 
* `ifndef       变量名`        


表达式的写法
* 括号  `(arg1, arg2)`
* 单引号 `'arg1' 'arg2'`
* 双引号 `"arg1" "arg2"`
* 两个变量分别 单双引号 `"arg1" 'arg2'`  `'arg1' "arg2"`


变量名 (name of a variable)  和 变量引用 ( reference to a variable)
* 变量名仅仅是 定义一个变量时候变量的名称   `MY_VAR`
* 变量引用则是一个表达式用于 获取某个变量的值 `$(MY_VAR)`
* ifdef ifndef 的正确用法是用于验证某个变量是否被定义过, 因此不要传入变量引用  

```makefile
bar =
foo = $(bar)
# 传入的是 foo 变量名称, 因为 foo=$(bar) 已经被定义过, 因此执行 true 的语句
ifdef foo
# 执行的部分
frobozz = yes
else
frobozz = no
endif


foo =
# foo 没有被定义 (被取消定义了) , 因此会执行 false 的代码 
ifdef foo
frobozz = yes
else
frobozz = no
endif



```

## 7.3. Conditionals that Test Flags - 

# 8. Functions for Transforming Text

make 本身提供了可以用于 makefile 里的许多函数, 这些函数大部分都是用来处理 字符串的, 即用于处理各种文件名或者路径  

同时, 用户也可以自定义各种函数  


## 8.1. Function Call Syntax

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


## 8.2. Functions for String Substitution and Analysis

主要用于字符串处理的一些函数  

索引:
* filter                : 过滤掉不满足 pattern 的, removing any words that do not match

### 8.2.1. 字符串替换

* `$(subst from,to,text)`
* `$(patsubst pattern,replacement,text)`
  

### 8.2.2. 字符串调整

* `$(strip string)`
* `$(findstring find,in)`
* `$(filter pattern…,text)`

### 8.2.3. 值列表调整

* `$(sort list)`
* `$(word n,text)`
* `$(firstword names…)`
* `$(lastword names…)`


## 8.3. Functions for File Names
<!-- 完 -->

专门用于针对文件的路径进行处理的函数, 包括分离文件名的各个部分等  

输入的参数被视为通过空格分割的一系列文件名, 同时输出也是同理  


提取函数
* `$(dir names…)`       : 提取文件夹路径, 即最后一个 `/` 之前的部分, 如果没有 `/`, 则文件夹路径视作 `./`
* `$(notdir names…)`    : 提取非文件夹路径的部分, 即最后一个 `/` 之后的部分, 如果没有 `/` 则不做任何更改
* `$(suffix names…)`    : 提取后缀, 如果文件没有后缀则返回空, 即返回的数值个数可能会变少
* `$(basename names…)`  : 提取非后缀的部分, 同理如果没有后缀则不更改

示例: 执行结果
* `$(dir src/foo.c hacks)`   `src/ ./` 
* `$(notdir src/foo.c hacks)`  `foo.c hacks`
* `$(suffix src/foo.c src-1.0/bar.c hacks)`  `.c .c`
* `$(basename src/foo.c src-1.0/bar hacks)`  `src/foo src-1.0/bar hacks`


路径修改函数
* `$(addsuffix suffix,names…)`          : 追加后缀
* `$(addprefix prefix,names…)`          : 追加前缀
* `$(join list1,list2)`                 : 两个 list 的元素依次结合, 如果元素个数不匹配, 尾部剩余的部分则会原样拷贝  

示例 : 执行结果
* `$(addsuffix .c,foo bar)`     `foo.c bar.c`
* `$(addprefix src/,foo bar)`   `src/foo src/bar`
* `$(join a b,.c .o)`           `a.c b.o`


文件检索: 
* `$(wildcard pattern)`                  : 通配符手动调用函数
  * 详细的通配符使用方法需要查阅 章节 4.4 Using Wildcard Characters in File Names

路径转换:
* `$(realpath names…)`                   : 路径转换, 包括转换链接, 消除 `../`, 消除重复的 `/`, 验证路径是否存在, 如果转换失败则返回空字符串
* `$(abspath names…)`                    : 有些类似于 realpath, 但不进行验证存在, 同时不进行链接转换  

## 8.4. Functions That Control Make - 控制 make 执行的函数  

包括了 3 个函数, 主要用于某些环境错误被检测到的时候 stop make

* `$(error text...)`
  * 用于处理严重 error, 此时 make 会被停止, 同时输出 text
  * 注意, 报错会发生在 `error` 函数被计算的时候, 因此如果 error 函数如果被定义在 recipe 或者 variable 赋值的右边, 那么直到 error 被正式计算为止都不会报错  
  * 函数会返回对应的 text, 但是一般没什么用处
* `$(warning  text...)`
  * 与 error 类似, 但是不会退出 make
  * 函数返回值为 空
* `$(info  text...)`
  * 仅仅是打印信息, 不做其他任何事





# 9. How to Run make - make CLI 文档


记载了关于 make CLI 的相关使用方法  

make 程序本身的退出代码书写在了章节开头 
* 0     : 成功运行
* 2     : 遇到错误, 同时会打印相关的错误描述信息
* 1     : 与 CLi 中的 `-q` 标志相关, 在 make 确认某些目标尚未更新的情况下


* `-j [N], --jobs[=N]`
    * Allow N jobs at once; infineite jobs with no arg.
    * 允许并行编译
    * 不指定数字的时候自动探测最大的可用CPU


## 9.1. Arguments to Specify the Makefile

<!-- 完 -->
通过 `-f` or `--file` 参数, 指定要运行的 makefile 文件.  

通过多次调用 `-f` 参数, 可以传入多个 makefile 文件, 同时各个文件会 joint 成为最终运行的文件  

默认情况下运行的文件名按顺序为 `GNUmakefile` `makefile` `Makefile`

## 9.2. Arguments to Specify the Goals
<!-- 完 -->
编译目标  goals , 从逻辑上来说, 本次运行 make 所要达成的最终的更新目标  

默认下, make 会运行所定义的 第一个 `非点号开头的` target, 因此约定俗成中, 第一个 target 总是用来编译整个程序.   

可以通过特殊变量 `.DEFAULT_GOAL` 来更改默认的编译目标  

target 可以被指定为由隐式规则生成的, 因此即使在 makefile 中没有被显式的定义也可以被指定.  


在 CLI 中, 可以传递一个或者多个 target 用于指定要执行的目标, 多个 target 会由 make 根据 `the order you name them` 的顺序来实行
* 通过 CLI 传入的 target 会作为一个特殊变量 `MAKECMDGOALS`, 如果没传入 则该变量为空. 注意该变量只能在特殊情况下使用, 不要轻易调用
* 一种可选的`MAKECMDGOALS`使用方法是, 通过检测 GOALS 是不是 clean, 来避免一些不必要的生成再立即删除的操作  


CLI 中可以显式的让 make 执行 Phony Target, 以下是典型的 Phony 以及 empty target names, 约定俗成下会经常使用它们. 
* `all`   : 执行所有 makefile top-level targets
* `clean` : 删除所有 由运行 make 所生成的文件
* `mostlyclean`   : 用于避免一些可能并不需要或者不想重新编译的文件, 例如 libgcc.a
* `distclean`, `realclean`, `clobber` : 这几个 target 一般用来指代会删除比 clean 更多的文件, 例如一些用于编译的由用户创建的配置文件等
* `install`     : linux 系统意义上的安装, 即复制各种可执行文件到用户搜索目录下
* `print`       : 打印 listings of the source files that have changed.
* `tar`         : 打包源文件, 创建 tar 文件
* `shar`        : 创建一个 shell archive (shar) 的源文件打包
* `dist`        : 更加广义上的打包用于源码发布, 可以是 tar 或者 shar 或者其他的方式
* `TAGS`        : 更新 tags table 
* `check`, `test`  : 运行一些该项目的测试程序  

同时作为 GNU 软件的话还有另外一个 list 代表所有 GNU 软件所必须定义的 target, 参阅 `Standard Targets for Users`


## 9.3. Overriding Variables

<!-- 完 -->

在 CLI 中, 通过 `v=x` 的方式可以传入变量, 这种情况下变量会覆盖掉 makefile 中所定义的具体的值.  

例如: makefile 中定义 `CFLAGS=-g` 则在 CLI 中可以传入 `make CFLAGS='-g -O'` 用以实现编译的不同动作.  

同理, simply-expanded variable 也可作为 CLI 变量被传入, 此时使用的是 `:=` 而不是 `=`


# 10. Using Implicit Rules

You can define your own implicit rules by writing `pattern rules`.    
`Suffix rules` are a more limited way to define implicit rules.  
`Pattern rules` are more general and clearer, but suffix rules are retained for compatibility. 

## 10.1. Defining and Redefining Pattern Rules

通过定义 Pattern rule 可以实现对 implicit rule 进行定义或重定义  

Pattern rule: 从形式上和 ordinary rule 没有区别, 但是包含了 `%` 符号  

Pattern rule 的 `%` 扩展发生在 变量替换和 函数执行 之后

### 10.1.1. Introduction to Pattern Rules

一个 rule 被称为 pattern rule 是因为它的 target 里包含一个 `%`, 可以匹配任何 nonempty substring, 被 match 的部分被叫做 `stem`  
A pattern rule contains the character ‘%’ (exactly one of them) in the target


`%` 的匹配替换是以 target 为主的, 即 `%` 也可以出现在 prerequisite 里, 但是 stem 的值是依据 target 的


### 10.1.2. Automatic Variables

让 makefile 变成天书的罪魁祸首, `automatic variables` 根据每一项 rule 来重新计算该 variable 的值  

automatic variable 仅仅只在 recipe 里有效, 让 prerequisite list 里也能使用 `automatic variables` 的特性称为 `Secondary Expansion` 在之前的章节里介绍

对于存在的一个段 `all: library.cpp main.cpp`  
* `$@`          : 本段的`target`  即 `all`, 对于具有多个目标的 pattern rule, `$@` 也是根据 rule 来确定要运行对应 recipe 时候的 target
* `$<`          : `library.cpp`   即第一个 prerequisite, 即使 prerequisite 是根据 implicit 来自动添加的, 也是指的是第一个 
* `$^`          : library.cpp 和 main.cpp  即**所有的** dependencies
* `$?`          : 本次执行中 比 target 新的所有 `prerequisites`, 如果 target 不存在, 则认为所有的 prerequisites 都是新的. 该变量在用于更新 archive lib 的时候很有用
* `$*`          : 

我们的 dependencies 中的内容, 往往和 g++ 命令中的内容重复,例如: 
```makefile
hello: main.o function1.o function2.o
        $(CC) $(LFLAGS) main.o function1.o function2.o -o hello

# 使用特殊符号简化后
hello: main.o function1.o function2.o
        $(CC) $(LFLAGS) $^ -o $@        
```


# 11. Using make to Update Archive Files

使用 make 来对 archive 进行自动化更新

