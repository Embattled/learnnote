# 1. CMake

CMake 是一个跨平台的工具, 用于 `描述安装和编译过程`
* 输出各种各样的 makefile 或者 project 文件
* CMake 并不直接建构出最终的软件(不进行代码编译本身), 而是产生标准的建构档


CMake的使用流程
1. 编写CMake编译描述文件 `CMakeLists.txt`
2. `cmake <CMakeLists.txt的目录>` 会在当前目录下生成 CMake 的工程文件
    * 包括对应的 `CMakeFiles` 目录
    * 可以用于 make 程序的 `MakeFile`
    * 各种 Cache
3. `cmake --build <工程目录>`


CMake工具包: CMake 作为一个组件, 除了用于编译项目的 CMake 以外, 还有其他工具
* cmake
* ctest
* cpack

# 2. Tutorial

## 2.1. Basic Starting Point

* CMake 的核心描述文件是 `CMakeLists.txt`, 用纯文本描述
* CMake 是大小写不敏感的 支持大小写混合

```MakeFile
# 所有的 CMakeLists 都必须以该函数开头, 用于指定 CMake 的最低版本
cmake_minimum_required(VERSION 3.10)

# set the project name, 用于指定各种项目的全局信息, 例如项目名称, 版本号等  
project(Tutorial)
project(Tutorial VERSION 1.0)

# 设置一些基本的全局变量, 例如C++版本
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 添加一个配置文件, 会在编译的时候拷贝到源文件目录中并进行值替换
configure_file(TutorialConfig.h.in TutorialConfig.h)

# add the executable. 用于添加要编译出的可执行文件  
add_executable(Tutorial tutorial.cxx)

# 为要编译出的可执行文件提供 包含目录
target_include_directories(Tutorial PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           )
```

## 2.2. Basic Starting Point


# 3. Command-Line Tools


# 4. cmake-commands

## 4.1. CMake Scripting Commands

These commands are always available.
CMake脚本文件函数, 用以进行各种文件操作, 可能会在 CMake 文件中出现多次

### 4.1.1. find 系列

用于在系统或者工作目录中查找各种环境条件

#### 4.1.1.1. find_package

Find a package (usually provided by something external to the project)
是CMake 用于寻找第三方库的主要方法, [相关Guide](https://cmake.org/cmake/help/latest/guide/using-dependencies/index.html) 里面就着重介绍了该 Command.

该功能很庞大, 文档内部也分了许多章节, 同时作为一个接口该功能有两种模式, Module mode 和 Config mode

大部分库都支持 CMake 的检索, 即实现了一些标识信息用于 CMake 

整体上:
* 执行该接口后, 名为 `<PackageName>_FOUND` 的变量会被赋值, 该变量主要用于输入 if 进行逻辑分支  

一般情况下, projects should generally use the Basic Signature , 这里也优先只学习该章节

##### 4.1.1.1.1. Basic Signature

```makefile
find_package(<PackageName> [version] [EXACT] [QUIET] [MODULE]
             [REQUIRED] [[COMPONENTS] [components...]]
             [OPTIONAL_COMPONENTS components...]
             [REGISTRY_VIEW  (64|32|64_32|32_64|HOST|TARGET|BOTH)]
             [GLOBAL]
             [NO_POLICY_SCOPE]
             [BYPASS_PROVIDER])
```
即 find_package 的基本语法构成, 同时支持两种 mode. 具体区分在 `[MODULE]` 参数上.  



参数上:
* `[version]`     : 用于指定包的版本, 目前新版支持两种格式
  * A single version with the format `major[.minor[.patch[.tweak]]]`, where each component is a numeric value.
  * 版本区间, 只有 CMake 3.19 才支持.
* `[EXACT]`       : 用于进一步修饰 version 设定, 要求版本号必须完整匹配  
* `[QUIET]`       : 安静执行, 主要用于 非 REQUIRED 的包在没有找到的时候的提示信息  
* `[REQUIRED]`    : 用于声明该包是必须的, 如果未查找到则会报错并终止 CMake build
* `[COMPONENTS]`  : 用于指定该包的依赖组件, 包的依赖包, 如果某一个组件不满足, 则整个包都会被返回为 not found
  * 为了便于书写 当 `REQUIRED` 出现的时候 `components` 可以直接写在之后, 而无需书写 `COMPONENTS` 关键字
* `[OPTIONAL_COMPONENTS components...]` : 包的可选的依赖包, 似乎并没有那么重要, 可能只是为了获取一些信息
  




### 4.1.2. configure_file

```makefile
configure_file(<input> <output>
               [NO_SOURCE_PERMISSIONS | USE_SOURCE_PERMISSIONS |
                FILE_PERMISSIONS <permissions>...]
               [COPYONLY] [ESCAPE_QUOTES] [@ONLY]
               [NEWLINE_STYLE [UNIX|DOS|WIN32|LF|CRLF] ])
```

功能:
* 将 input 复制到 output, 将一些与 CMake 进行绑定的具有配置意义的代码复制到源文件目录中  
* 在复制的过程中, 将 CMake 语法 定义的特殊变量进行值替换, 例如
  * `@VAR@`
  * `${VAR}`
  * `$CACHE{VAR}`
  * `$ENV{VAR}`
  * 如果在 CMake 文件中这些变量没有被定义, 则会被替换成空字符串  

进行值替换的语法也在该 command 的文档中书写了 TODO


### 4.1.3. list

Operations on semicolon-separated lists. 对 list 进行各种各样的操作, list 本身是以分号来进行分割的


### 4.1.4. set - CMake 文件中的变量

CMake 中非常基本的为一个变量赋值的 command, 具体的可以应用于 normal, ache, environment variable 值  
关于作用域的详情需要参考 cmake-language variables

在 set 的文档中, 占位符 `<value>` 用于指代 0个或以上的参数, 其中1个以上的参数会自动变为 `a semicolon-separated list`

#### 4.1.4.1. Set Normal Variable

设置一个 CMake 基本变量
`set(<variable> <value>... [PARENT_SCOPE])`

如果一个以上的 `<value>` 被输入, 则设置该变量, 否则 0 参数的适合表示取消设置该变量, 这种情况下的操作与 `unset(<variable>)` 的行为相同  

#### 4.1.4.2. Set Cache Entry

TODO

#### 4.1.4.3. Set Environment Variable

设置环境变量  
`set(ENV{<variable>} [<value>]`  

设置一个变量到环境变量空间, 这并不是实际的影响到 bash 环境中, 而是用于接下来的 `$ENV{<variable>}$` 调用  
* 只会影响当前的 CMake 进程
* 不会影响
  * 调用 CMake 的母进程
  * 系统环境

同样的, 不传入 `<value>` 则会清空该环境变量. 而环境变量不支持多 value 参数, 因此首个 value 之后的参数输入会被忽略  



## 4.2. CMake Project Commands

These commands are available only in CMake projects.  
对项目整体进行各种参数配置

在 `CMakeLists.txt` 中, 对项目的各种配置是以函数形式存在的

### 4.2.1. project

```makefile
project(<PROJECT-NAME> [<language-name>...])
project(<PROJECT-NAME>
        [VERSION <major>[.<minor>[.<patch>[.<tweak>]]]]
        [DESCRIPTION <project-description-string>]
        [HOMEPAGE_URL <url-string>]
        [LANGUAGES <language-name>...])
```

用于对项目进行最基础的描述, 是构建 CMake 项目的基础函数之一
* PROJECT-NAME          : 项目的名称, 在`CMakeLists.txt` 中调用该函数的时候, 该名称会存储在 `CMAKE_PROJECT_NAME` 中 
  * 同时会配置其他几个目录变量, 可以方便的在之后的编辑中直接调用
  * `PROJECT_SOURCE_DIR`, `<PROJECT-NAME>_SOURCE_DIR`
  * `PROJECT_BINARY_DIR`, `<PROJECT-NAME>_BINARY_DIR`
  * `PROJECT_IS_TOP_LEVEL`, `<PROJECT-NAME>_IS_TOP_LEVEL`  New in version 3.21.


### 4.2.2. add - 与添加相关的系列指令

* add_executable      : 添加可执行文件输出
* add_library         : 添加库文件输出
* add_subdirectory    : 为整个项目添加子目录

#### 4.2.2.1. add_executable

为项目添加一个编译输出可执行文件 target 

```makefile
add_executable(<name> [WIN32] [MACOSX_BUNDLE]
               [EXCLUDE_FROM_ALL]
               [source1] [source2 ...])
```
* name : 用以定义该 target 的逻辑名称, 需要全局唯一, 同时也是输出文件名 (不包括后缀名部分)
* source : 该可执行文件的输入代码,  3.1 版本开始该部分可以是一个生成表达式
  * 可以用 `target_sources()` 来单独添加源文件, 在这里就可以省略了 

#### 4.2.2.2. add_library

为一个项目添加一个使用指定源文件编译的库


##### 4.2.2.2.1. Normal Libraries
```makefile
add_library(<name> [STATIC | SHARED | MODULE]
            [EXCLUDE_FROM_ALL]
            [<source>...])
```

* `<name>` : 为该 library 的逻辑名称, 需要全局唯一, 同时会根据平台来决定最终的实际输出文件名 (such as `lib<name>.a` or `<name>.lib`)
* `[STATIC | SHARED | MODULE]`  : 用于指定该库的编译模式
  * STATIC : 即静态库, 用于在编译的最终阶段链接到输出的可执行文件里
  * SHARED : 动态链接库
  * MODULE : plugins that are not linked into other targets but may be loaded dynamically at runtime using dlopen-like functionality

##### 4.2.2.2.2. Imported Libraries

#### 4.2.2.3. add_subdirectory

为当前 build 添加一个子目录, 子目录中要有子目录自己的 CMakeLists.txt  

在指定该 command 的时候会立即执行子目录的 CMakeLists.txt, 在完成后才会返回继续执行主文件

```makefile
add_subdirectory(source_dir [binary_dir] [EXCLUDE_FROM_ALL] [SYSTEM])
```
* `source_dir ` : the directory in which the source `CMakeLists.txt` and code files are located
  * 可以是绝对路径, 但是更多的情况是相对路径
* `binary_dir ` : 用于指定该目录下的编译输出路径
  * 默认情况下使用 `source_dir`
  


### 4.2.3. target - 与编译的 target 相关的系列命令

#### 4.2.3.1. target_include_directories

```makefile
target_include_directories(<target> [SYSTEM] [AFTER|BEFORE]
  <INTERFACE|PUBLIC|PRIVATE> [items1...]
  [<INTERFACE|PUBLIC|PRIVATE> [items2...] ...])
```

对指定的 target 添加编译时候的 include 目录
* target : 必须是之前由 `add_executable()` or `add_library()` 所添加的, 同时不能是 `ALIAS target`
* `[AFTER|BEFORE]` : 设置该 include 的位置, 可以是 appending 或者 prepending
* `<INTERFACE|PUBLIC|PRIVATE>` : 这是一个必要参数, 该关键字用于定义接下来的 `items` 的 scope 属性, 具体为:
  * `PRIVATE` and `PUBLIC` items will populate the `INCLUDE_DIRECTORIES` property of `<target>`
  * `PUBLIC` and `INTERFACE` items will populate the `INTERFACE_INCLUDE_DIRECTORIES` property of `<target>`

#### 4.2.3.2. target_link_directories

#### 4.2.3.3. target_link_libraries

为 target 的编译添加一个 library, target 的编译条件只会影响自身, 而 library 的编译条件会传播到 target  



### 4.2.4. set

```makefile
set(<variable> <value>... [PARENT_SCOPE])
```
设置对应的变量, 可以用于配置项目参数


# 5. cmake-language

CMake 文档中用于书写语法的方式很有意思, 使用正则表达式的方式来书写一个语法  

## 5.1. Syntax

CMake 语言的编码 (Encoding): 一个 CMake Language source file 应该以 7-bit ASCII 字符来书写用以实现最大的兼容性. 换行符可以以 `\n` 或者 `\r\n` 来书写  

CMake 的源文件的构成:
* zero or more Command Invocations separated by 
  * newlines
  * optionally spaces
  * Comments

### 5.1.1. Command Invocations 命令调用

命令调用是一种语法, 表现为 一个名称, 后接一个括号, 括号里面是用空格分割的参数

用正则表达式可以描述为 :   
```sh
command_invocation  ::=  space* identifier space* '(' arguments ')'
identifier          ::=  <match '[A-Za-z_][A-Za-z0-9_]*'>
arguments           ::=  argument? separated_arguments*
separated_arguments ::=  separation+ argument? |
                         separation* '(' arguments ')'
separation          ::=  space | line_ending
```

CMake 中的 Command names 不区分大小写, 同时 `未加引号的括号` 必须平衡

### 5.1.2. Command Arguments

即 Command Invocations 语法中的 arguments   

CMake 对于 Arguments 细分了3种类型  
`argument ::=  bracket_argument | quoted_argument | unquoted_argument`






### 5.1.3. Comments 注释

注释以 `#` 开头
* 并且 not inside a 
  * Bracket Argument
  * Quoted Argument
* not escaped with \ as part of an Unquoted Argument

CMake 源文件支持两种注释  
* Bracket Comment 
* Line Comment


## 5.2. lists

从概念上, 所有的数值在 CMake 语言中都以 strings 来保存, 但是一个 string 也可能作为一个 list 来处理.


list 从存储上来说, 是将一个 list 的元素以 `;` 分割, 保存为一整个连续的字符串, 因此 list 更多的适用于简单的使用场景, 不支持 list 的嵌套, 因为 元素中的 `;` 并不会自动的被转义


```makefile
# 对于最经典的 set 命令, 支持多个元素的输入
set(srcs a.c b.c c.c)     # sets "srcs" to "a.c;b.c;c.c"
set(x a "b;c")            # sets "x" to "a;b;c", not "a;b\;c"

```

# 6. cmake-variables

variables that are provided by CMake or have meaning to CMake when set by project code.

CMake reserves identifiers that:
* begin with `CMAKE_` (upper-, lower-, or mixed-case), or
* begin with `_CMAKE_` (upper-, lower-, or mixed-case), or
* begin with `_` followed by the name of any CMake Command.

类似于编程语言的变量, 可以用来方便的书写 cmake 配置文件, 一些 CMake 命令会自动的定义一些变量

在脚本中通过 `${Var}` 来动态的使用变量, 来替换一些路径/文件名



