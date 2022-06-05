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

## 1.1. CMakeLists.txt

* CMake 的核心描述文件是 `CMakeLists.txt`, 用纯文本描述
* CMake 是大小写不敏感的 支持大小写混合

```MakeFile
cmake_minimum_required(VERSION 3.10)

# set the project name
project(Tutorial)

# add the executable
add_executable(Tutorial tutorial.cxx)
```

# 2. cmake-commands

## 2.1. CMake Scripting Commands

These commands are always available.
CMake脚本文件函数, 用以进行各种文件操作, 可能会在 CMake 文件中出现多次

### 2.1.1. configure_file

```makefile
configure_file(<input> <output>
               [NO_SOURCE_PERMISSIONS | USE_SOURCE_PERMISSIONS |
                FILE_PERMISSIONS <permissions>...]
               [COPYONLY] [ESCAPE_QUOTES] [@ONLY]
               [NEWLINE_STYLE [UNIX|DOS|WIN32|LF|CRLF] ])
```

功能:
* 将 input 复制到 output


## 2.2. CMake Project Commands

These commands are available only in CMake projects.  
对项目整体进行各种参数配置

在 `CMakeLists.txt` 中, 对项目的各种配置是以函数形式存在的

### 2.2.1. project

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
  * 同时会配置其他几个目录变量
  * `PROJECT_SOURCE_DIR`, `<PROJECT-NAME>_SOURCE_DIR`
  * `PROJECT_BINARY_DIR`, `<PROJECT-NAME>_BINARY_DIR`
  * `PROJECT_IS_TOP_LEVEL`, `<PROJECT-NAME>_IS_TOP_LEVEL`  New in version 3.21.


### 2.2.2. add_executable

```makefile
add_executable(<name> [WIN32] [MACOSX_BUNDLE]
               [EXCLUDE_FROM_ALL]
               [source1] [source2 ...])
```
* 为项目添加一个编译输出可执行文件 target 
* name : 用以定义该 target 的逻辑名称, 需要全局唯一, 同时也是输出文件名 (不包括后缀名部分)
* source : 该可执行文件的输入代码,  3.1 版本开始该部分可以是一个生成表达式
  * 可以用 `target_sources()` 来单独添加源文件, 在这里就可以省略了 


### 2.2.3. target_include_directories

```makefile
target_include_directories(<target> [SYSTEM] [AFTER|BEFORE]
  <INTERFACE|PUBLIC|PRIVATE> [items1...]
  [<INTERFACE|PUBLIC|PRIVATE> [items2...] ...])
```

对指定的 target 添加编译时候的目录
* target : 必须是之前由 `add_executable()` or `add_library()` 所添加的, 不能是 ALIAS target


### set

```makefile
set(<variable> <value>... [PARENT_SCOPE])
```
设置对应的变量, 可以用于配置项目参数


# 3. cmake-variables

variables that are provided by CMake or have meaning to CMake when set by project code.

CMake reserves identifiers that:
* begin with `CMAKE_` (upper-, lower-, or mixed-case), or
* begin with `_CMAKE_` (upper-, lower-, or mixed-case), or
* begin with `_` followed by the name of any CMake Command.

类似于编程语言的变量, 可以用来方便的书写 cmake 配置文件, 一些 CMake 命令会自动的定义一些变量

在脚本中通过 `${Var}` 来动态的使用变量, 来替换一些路径/文件名



