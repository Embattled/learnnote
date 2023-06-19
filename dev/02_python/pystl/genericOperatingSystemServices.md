# 1. Generic Operating System Services

* 和 runtime 中的 python 解释器模块不同, 这部分的模块更贴近操作系统方面
* 在几乎所有的操作系统都能使用, 提供了与操作系统相关的所有操作的接口, 主要都是基于 C 接口实现的  

杂项系统库:
* os        各种基础操作啥都有
  * os.path 作为 os 库中作用比较几种的子集, 放在了 path 专门的文档中  

专用目的库:
* argparse  : 专门用来处理 python 命令行参数的
* getpass   : 专门用来处理密码输入的
* ctypes    :

# 2. os

* os — Miscellaneous operating system interfaces, 包含了各种操作系统相关的方法, 是一个杂模组
* 里面很多函数都是限定操作系统的
* 非常大的库
* 经常与 sys 库搞混
* 该库更加贴近于不与 python 相关的 os 的功能


* 一些非常常用的功能被移动到了python内部函数:
  * 如果想使用文件IO, 查看内部函数 `open()`即可, 它是 `os.fdopen()` 的alias
* 一些目的比较明确的库, 被 python 封装成了其他专用的 STL 包
  * 管理运行时 paths, 使用模组 `os.path`
  * 文件读取的相关内容在 `fileinput` 中
  * 临时文件          `tempfile`
  * 高级路径文件操作  `shutil`


## 2.1. Files and Directories

os.mkdir(path, mode=0o777, *, dir_fd=None)

os.makedirs(name, mode=0o777, exist_ok=False)

### 2.1.1. os.walk

- `os.walk(top, topdown=True, onerror=None, followlinks=False)`
- 遍历一个文件目录下的所有子目录(包括该文件目录本身), 一直遍历到没有子目录
- 返回一个三元元组 `3-tuple (dirpath, dirnames, filenames).`
  - dirpath : 返回目录
  - dirnames: 返回该目录下的 所有子目录名称 列表形式
  - filenames:返回该目录下的 所有文件名称   列表形式
- 参数:
  - top     : 搜索的根目录
  - topdown : 是否自顶向下搜索
  - floowlink:是否进入目录中的软连接进行搜索, 可能会导致无限递归

```py
import os
from os.path import join, getsize
for root, dirs, files in os.walk('/home/eugene/workspace/learnnote/cvml'):
    print(root, "consumes", end=" ")
    print(sum(getsize(join(root, name)) for name in files), end=" ")
    print("bytes in", len(files), "non-directory files")

```

## 2.2. Process Parameters 进程参数

These functions and data items provide information and operate on the current process and user.  

可以提取一些调用该 python 程序的进程信息和用户信息


### 2.2.1. 各种 get

分类
* get     :
* gete    : 


Unix 专属:
* `os.ctermid()`    : Return the `filename` corresponding to the controlling terminal of the process.
* `os.getegid()`    : effective group id of the current process.
* `os.getuid()`     : Return the current process’s real user id.
* `os.geteuid()`    : current process’s effective user id.
* `os.getgid()`     : Return the real group id of the current process.


Unix. Windows:
* `os.getlogin()`   : Return the name of the user logged in on the controlling terminal of the process.
  * 不被官方推荐使用, 应该用 `getpass.getuser()` 代替

### 2.2.2. 环境变量管理

* `os.environ` A mapping object where keys and values are strings that represent the process environment.
  * 不是函数, 而是一个具体的字典对象. eg `environ['HOME']`
  * 该对象不仅可以用来查询, 修改该字典会自动写回环境变量到系统 (通过调用 ` os.putenv(key, value)¶` )

## 2.3. Process Management

These functions may be used to create and manage processes. 可以用来通过Python解释器进程再创建各种子进程


### 2.3.1. os.system

是 python3 STL 的 subprocess 重点想要代替的基础函数

Execute the command (a string) in a subshell:
* 是基于 C 语言的系统 API `system()` 实现的, 所以有相同的限制
* 对于更改解释器的标准输入流 `sys.stdin` 不会影响到该子进程
* 对于输出流, If command generates any output, it will be sent to the interpreter standard output stream.
* 返回值是该命令的执行结果
  * On Unix, the return value is the exit status of the process encoded in the format specified for wait().
  * On Windows, the return value is that returned by the system shell after running command. 

`os.system(command)`


# 3. time

* 与系统时间相关的函数, 大部分都是直接调取操作系统内置的 同名C函数  
* Python 中与时间操作的 module 还有 `datetime` 和 `calendar`

并不是所有的函数都可以在所有平台使用, 因为大部分函数都是直接调用 C 函数库

datetime: 属于 Datatype 类型, 不在该文件中 


Each operating system implements clocks and performance counters differently, and it is useful to know exactly which function is used and some properties of the clock like its resolution.   
The `time.get_clock_info()` function gives access to all available information about each Python time function.  
* `time.get_clock_info(name)`: 
  * 'monotonic': time.monotonic()
  * 'perf_counter': time.perf_counter()
  * 'process_time': time.process_time()
  * 'thread_time': time.thread_time()
  * 'time': time.time()

  
## 3.1. 相关定义

* epoch 时间是从 1970, 00:00:00 (UTC) 开始的, 根据平台不同可能会有不同的开始时间
* 该模组中的函数都是 C 相关的, 因此对日期的处理只限于 32位描述, 即 1970~2038
* 2位数的年份表示: `POSIX and ISO C standards` 69~99表示 `19**`, 0~68表示`20**`

| From                      | To                        | Use               |
| ------------------------- | ------------------------- | ----------------- |
| seconds since the epoch   | struct_time in UTC        | gmtime()          |
| seconds since the epoch   | struct_time in local time | localtime()       |
| struct_time in UTC        | seconds since the epoch   | calendar.timegm() |
| struct_time in local time | seconds since the epoch   | mktime()          |

```py
# 获取该计算机平台的时间表示开始时间
time.gmtime(0)

```
## 3.2. 获取时间

`time.clock()` 的不足之处
* 系统相关的
* Windows下会包括程序睡眠期间经过的时间, Unix 不包括
* 在 Windows 下有很高的精度, 但是 Unix 下精度很糟糕


普通时钟:  seconds since the epoch
* `time.time() → float`
* `time.time_ns() → int`  version 3.7


单调时钟 : The clock is not affected by system clock updates.
* 存在意义: 主要用于服务系统调度和超时检测, 直接使用系统时间会因为系统时间的更新导致程序出错
* 单次调用没有任何实际意义, 只有两次调用的差值才有意义
* `time.monotonic()->float` :  version 3.3
  * 返回 秒 单位的浮点时间, 
* `time.monotonic_ns()->int`:  version 3.7
  * 返回 ns 单位的整数时间

性能时钟 :  `system-wide`
* 功能: 主要解决的 `time.clock()` 的几种缺点, 保证在不同系统下都是最高精度的时间测量器
* 单次调用没有任何实际意义, 只有两次调用的差值才有意义
* include time elapsed during sleep and is system-wide
* `time.perf_counter() → float` 浮点返回值
* `time.perf_counter_ns() → int` 整数返回值

进程CPU时钟 : `process-wide by definition`
* sum of the `system and user CPU time` of the current `process`
* 注意该时间不是真实意义上的时间
* always measures CPU time, does not include time elapsed during sleep.
* has the best available resolution.
* 单次调用没有任何实际意义, 只有两次调用的差值才有意义
* `time.process_time() → float`
* `time.process_time_ns() → int`


线程CPU时钟:  thread-specific by definition
* sum of the `system and user CPU time` of the current `thread`
* always measures CPU time, does not include time elapsed during sleep.
* 单次调用没有任何实际意义, 只有两次调用的差值才有意义
* `time.thread_time() → float`
* `time.thread_time_ns() → int`


## 3.3. 时间格式转换

转成秒数
* `time.mktime(t)`
  * 输入 :  `struct_time` or full 9-tuple (since the dst flag is needed) 
    * `-1` as the dst flag if it is unknown
  * return : a floating point number, for compatibility with `time()`

转成 `struct_time`
* `time.gmtime([secs])`
  * 输入 :  a time expressed in seconds since the epoch
  * 默认值: `time()`
  * return : a `struct_time` in UTC in which the `dst flag is always zero`
* `time.localtime([secs])`
  * 输入和默认值都同上
  * dst flag is set to 1 when DST applies to the given time
* `time.strptime(string[, format])`
  * 输入 : Parse a string representing a time according to a format.
  * 根据指定的 format 将一个普通 string 转化成 struct_time
  * return :  `struct_time`

转成固定格式字符串
* `time.asctime([t])` :
  * 接受  : a tuple or `struct_time` as returned by `gmtime()` or `localtime()`
  * 默认值: `localtime()` , 即当前时间
  * return : 标准规格的时间字符串, 有固定长度 eg. `Sun Jun 20 23:21:05 1993`
* `time.ctime([secs])`:
  * 接受  : a time expressed in seconds since the epoch
  * 默认值: `time()`, 相当于 `asctime(localtime(secs))`
  * return : 标准规格的时间字符串, 同上

  
转成指定格式字符串
* `time.strftime(format, t=None )`
  * 接受 :  a tuple or `struct_time` as returned by `gmtime()` or `localtime()`
  * 默认值: `localtime()` , 即当前时间
  * return : 由 format 指定的字符串格式, 各种描述符如下
| 描述符 | 意义                  |
| ------ | --------------------- |
| %S     | 秒数                  |
| %M     | 分钟数                |
| %H %I  | 24-小时 12-小时       |
| %w     | 星期几 0表星期天      |
| %d     | 日期                  |
| %m     | 月份                  |
| %Y %y  | 四位数年份 两位数年份 |


## 3.4. 线程 CPU  (Unix限定)

* `time.pthread_getcpuclockid(thread_id)` :  version 3.7
  * 接受: thread_id , 必须是有效的 id, 否则会导致 段错误
  * return : 该线程对应的 CPU-time 的 clk_id
  * thread_id 可以通过 `threading.get_ident()` 来获得

* `time.clock_getres(clk_id)` :  version 3.3
  * 获取指定 clk_id 的 精度 (resolution)


* `time.clock_gettime(clk_id)->float` : version 3.3
  * 获取指定 clk_id 的时间
  * 返回值是浮点类型  
* `time.clock_gettime_ns(clk_id)->int` : version 3.7
  * 同上, 但是是整数返回值, 相对来说不会有精度损失
* `time.clock_settime(clk_id, time: float)` : version 3.3
  * float 类型设置指定 clk_id 的时间
  * 当前版本 clk_id 只能是 `CLOCK_REALTIME` 
* `time.clock_settime_ns(clk_id, time: int)` : version 3.7
  * 同上, 但是是 int 表示 ns 来设置


Clock ID Constants : used as parameters for `clock_getres()` and `clock_gettime()`

## 3.5. 获取规格化字符串时间



# 4. logging 日志模块


# 5. argparse

argparse组件可以很方便的写一个命令行界面, 可以很容易的定义程序的参数, 并从`sys.argv`中提取出来, 同时还会自动提供错误信息  

```py
import argparse
# 建立命令行翻译器, 可以同时设置该程序的主要目的
parser = argparse.ArgumentParser(description="calculate X to the power of Y")  
args = parser.parse_args() # 翻译传入的命令行参数
```

* `parser.parse_args()` 一般就是直接无参数调用, 会直接翻译传入的命令行参数, 返回一个 `Namespace` object
  * `Namespace` 就是定义在 argparse 包中的一个简单的类, 和字典类似, 但是 print()更加可读, 可以使用 `vars()` 方法进行转换成字典
  * `args.*` 用点号进行对应的参数访问

## 5.1. class argparse.ArgumentParser

整个库的核心类, 在 CLI 处理的一开始定义, 有很多参数   

```py
class argparse.ArgumentParser(
  prog=None, usage=None, description=None, 
  epilog=None, parents=[], 
  formatter_class=argparse.HelpFormatter, 
  prefix_chars='-', 
  fromfile_prefix_chars=None, 
  argument_default=None, 
  conflict_handler='error', add_help=True, 
  allow_abbrev=True, exit_on_error=True)
```

该类的所有参数都必须通过 kw 传递  
该类除了 `add_argument()` 的其他方法在 `Other utilities` 部分进行说明

说明用
* prog    : The name of the program `(default: os.path.basename(sys.argv[0]))`, 及在调用该程序的时候终端里应该输入的程序名称, 通过这个参数可以修改默认的程序名称

全局配置参数
* argument_default  : The global default value for arguments (default: None)
  * 用于全局的设置默认值
  * 除了默认的 None 以外, 另一个有用的是`argument_default=SUPPRESS`, 代表未出现的参数不会出现在返回的字典中 

组合:
* parents   :  A list of ArgumentParser objects whose arguments should also be included. 详情参考 subparser 的部分
* 


## 5.2. add_argument()

作为 ArgumentParser 的最关键的方法, 重要程度是相同的, 这里详细解释该方法的各个参数  

* 基础上使用 `parser.add_argument()` 来添加一个命令行参数  
* 添加可选参数的时候使用 `-` 或者 `--`, 会自动识别, 且在调用值的时候不需要加 `-`
  * 同时有 `-` 和 `--` 的时候会选择 `--` 的名字作为参数调用名

```py
ArgumentParser.add_argument(
  name or flags...
  [, action]      # 动作
  [, nargs]       # 多个参数
  [, const]
  [, default]     # 默认值无需多解释
  [, type]        # 转换的类型
  [, choices]     # 限制该参数可能的值, 输入一个 list 
  [, required]    # 官方推荐带 - 的就是可选参数, 否则必须参数, 官方doc说设置required会带来混淆因此应该避免使用. 但是实际上都在用
  [, help]        # str, 当使用 --help 时, 对该参数的说明
  [, metavar]
  [, dest])
```


按照常用的顺序进行说明:
1. `type`    指定该参数被转换成什么类型, 默认是 string, 这个可以指定为一个接受单个字符串的函数
```py
parser.add_argument('count', type=int)
parser.add_argument('distance', type=float)
parser.add_argument('street', type=ascii)
parser.add_argument('code_point', type=ord)

def hyphenated(string):
  pass
parser.add_argument('short_title', type=hyphenated)
```
2. `default`  就是默认值, 注意 `default=argparse.SUPPRESS` 表示为默认该成员不出现在返回中, 如果不指定的话默认值是 `None`
3. `action`   特殊操作, 代表该参数是一个动作参数, 默认是`store`, 在Action class 中具体说
4. `choices`  该参数只能是特定值中的某一项, 否则会报错, 用于防止输入非法的值


### 5.2.1. name or flags

```python

# action="store_true"
parser.add_argument("-V","--verbose", help="increase output verbosity", action="store_true")

# 用另一种方法增加信息冗余 action="count" ,统计某一参数出现了几次
# 可以识别 -v -vv -vvv '--verbosity --verbosity'
# 可以创建一个矛盾参数组, 当中的参数不能同时出现
group = parser.add_mutually_exclusive_group()
# 在group的方法中添加参数
group.add_argument("-v", "--verbose", action="store_true")
group.add_argument("-q", "--quiet", action="store_true")
```  


### 5.2.2. action - Action class

Action classes implement the Action API
* a callable which returns a callable which processes arguments from the command-line.
* Any object which follows this API may be passed as the action parameter to add_argument().

### 5.2.3. nargs

默认情况下, 单个命令行参数是与一个动作绑定的, 但是 nargs 定义可以将多个命令行参数绑定为同一个动作  

`nargs`支持的参数:
* N (an integer)    : 接下来的 N 个命令行参数会被打包成 list 作为一个参数值
  * 要注意当 N=1 的时候, 仍然会创建一个包含唯一参数的 list, 这与默认情况下仍然是不同的
* `'?'` : ? 字符    :  不是很好理解 `[TODO]`
* `'*'` : 星号字符   : 所有的 CLI 参数都会被打包成 list
  * 对于位置参数, 创建多个 `nargs='*'` 没有任何意义
  * 对于可选参数, 因为可以通过选项来进行区分所以可以多个存在, 但是在输入的时候要注意和位置星号参数的关系, 要先输入位置再输入可选

```py
# 两个可选和一个位置
parser = argparse.ArgumentParser()
parser.add_argument('--foo', nargs='*')
parser.add_argument('--bar', nargs='*')
parser.add_argument('baz', nargs='*')

# 在进行 parse 的时候需要注意参数的输入顺序
parser.parse_args('a b --foo x y --bar 1 2'.split())
Namespace(bar=['1', '2'], baz=['a', 'b'], foo=['x', 'y'])
```

### 5.2.4. type

参数的类型, 默认情况下 读取到的 CLI 参数都是作为 string 保存的, 然而某些情况下需要对 string 进行转换, 通过 type 可以便捷地对输入的值进行 转换和检查  

* 注意: 如果 type 参数和 default 参数一起使用, 那么类型转换只会在输入的值为默认值的时候生效  
* 可以调用的 type:  type 参数的输入可以是一个 callable, 即函数
  * 定义一个函数, 该函数接受一个 string, 并进行自定义的处理
  * 函数可以内建 ArgumentTypeError, TypeError, or ValueError, 这些 error 可以被正确的捕获并输出信息
* 基本上常用的 build-in 类型或者函数 都可以作为 type
* 对于复杂的类型, 例如 JSON 或者 YAML, 官方不推荐将内容的读取直接作为 callable type 来实现


```py
# 常用的 build-in 都可以作为 type, 
parser.add_argument('count', type=int)
parser.add_argument('distance', type=float)
parser.add_argument('street', type=ascii)
parser.add_argument('code_point', type=ord)
parser.add_argument('source_file', type=open)

# 官方不推荐这种 Type, 因为在其他参数不正确输入的情况下会导致文件没被正确的关闭
parser.add_argument('dest_file', type=argparse.FileType('w', encoding='latin-1'))

# 直接转化成 pathlib.Path
parser.add_argument('datapath', type=pathlib.Path)


# 自定义一个函数作为 type
def hyphenated(string):
    return '-'.join([word[:4] for word in string.casefold().split()])

_ = parser.add_argument('short_title', type=hyphenated)
parser.parse_args(['"The Tale of Two Cities"'])
# Namespace(short_title='"the-tale-of-two-citi')
```



## 5.3. 高级 args

除了基础的直接对 parser 里添加参数外, 还有其他特殊的参数类别  

* ArgumentParser.add_argument_group()
* ArgumentParser.add_mutually_exclusive_group
* ArgumentParser.add_subparsers
* ArgumentParser.set_defaults

### 5.3.1. Argument groups 参数分组

* 可以将不同的参数归到一组里,
* 这个组在参数调用的时候没有任何卵用
* 但是可以在程序帮助信息打印的时候更加清晰明确
  * `title` 和 `descripition` 参数用于清晰打印
  * 如果二者都为空, 那么在组之间只会有空行

```py
parser = argparse.ArgumentParser()
group = argparse.add_argument_group(title=None, description=None)
# 可以设置组的名称和说明 

group.add_argument('--foo', action='store_true')
group.add_argument('--bar', action='store_false')
# 添加至这个组的参数在打印参数说明的时候会单独分隔开, 方便区分
```

### 5.3.2. mutual exclusion 矛盾参数

* 可以将不同的参数归到一组里, 这个组的参数只能出现一种
* 每次调用该参数会返回一个类似于`ArgumentParser` 的对象, 通过调用该对象的 add_argument来实现矛盾参数
* required 参数代表该组参数是否必须出现一个, 同单个参数的 required 的差不多相同意思  

`ArgumentParser.add_mutually_exclusive_group(required=False)`  

```py
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()

group.add_argument('--foo', action='store_true')
group.add_argument('--bar', action='store_false')
```

### 5.3.3. subparsers 子解释器

```py
ArgumentParser.add_subparsers(
  [title][, description][, prog]
  [, parser_class][, action]
  [, option_strings][, dest]
  [, required][, help][, metavar])
```

许多程序会将其功能拆分成许多子命令
* eg. `git commit  git add`
* 子命令之间需要的参数并不相同
* 不希望各个功能的参数一直保存在 parser 里
* 通过拆分, 可以让各个子命令读取后返回的 Namespace 互相独立, 即字典里只包括 main 的参数以及命令行所选择的 subparser 里的参数
* 各个 subparser 里的参数名字是可以重复的, 完全独立
* 想要打印帮助信息的时候, 如果选择了 subparser, 那么 parent parser 的帮助信息是不会被打印的


* `add_subparsers()` 方法一般无参数调用, 返回一个专有的特殊对象
  * 该特殊对象只有一个方法 `add_parser()`
  * `add_parser()` 方法接受一个 `command name`, 以及完整的 `ArgumentParser` 构造函数参数, 可以多次调用创建不同的 `subparser`
  * 返回的 ArgumentParser 对象和基础的该类对象在使用上没有任何区别, 可以嵌套
  * 结合 `set_default()` 方法可以非常方便的实现功能选择模块 


`add_subparsers()` 方法尽管一般无参数调用, 但是还是说明下参数功能:
* 

```py


```
### 5.3.4. Parser defaults

`ArgumentParser.set_defaults(**kwargs)` 

* 该方法传入的内容都是直接作为字典数据传入 args
* allows some additional attributes that are determined without any inspection of the command line to be added.
* 另一种默认值的写法, 该默认值会覆盖 `add_argument()` 的效果, 但是不会覆盖命令行的输入

该方法可以设置一些无法通过命令行输入的参数, 例如设置函数, 和 subparser 结合可以实现自动执行对应函数的功能

```py
parser=argparse.ArgumentParser()
parser.set_defaults(default_func=func)

subparser=parser.add_subparsers()
parser1=subparser.add_parser('a1')
parser1.set_defaults(default_func=func1)
parser2=subparser.add_parser('a2')
parser2.set_defaults(default_func=func2)

args=parser.parse_args()

# 可以根据进入 subparser 自动选择正确的功能
args.default_func()
```

## 5.4. Parser defaults



# 6. getpass  — Portable password input 

类似于 argparse , 只是该模组只针对密码界面    
The getpass module provides two functions:
* `getpass.getpass(prompt='Password: ', stream=None)`
  * prompt : 提示信息
  * 可以用字符串存储得到的密码 `input_str=getpass.getpass()`
* `getpass.getuser()`
  * 获取当前进程的用户名
  * This function checks the environment variables `LOGNAME`, `USER`, `LNAME` and `USERNAME`, in order, and returns the value of the first one which is set to a `non-empty string`. 
  
# 7. ctypes — A foreign function library for Python

foreign function library for Python, 外部函数库
* 提供了与 C 语言兼容的数据类型
* 允许调用 DLLs 或者其他 Shared libraries 的函数
* 用于将库函数应用在纯 Python 中


## 7.1. 使用指南

1. 建立动态库示例
2. 获取动态库函数

### 7.1.1. 获取动态库实例


动态库默认实例, You load libraries by accessing them as attributes of these objects
* cdll      : standard cdecl calling convention
* windll    : stdcall calling convention
* oledll    : stdcall calling convention, and assumes the functions return a Windows `HRESULT` error code.
  * The error code is used to automatically raise an `OSError` exception when the function call fails.
* 可以按照访问成员的方式来访问动态库
```py
from ctypes import *
print(windll.kernel32)  
print(cdll.msvcrt)      
libc = cdll.msvcrt    
```

* 对于 Windows, Python 实现会自动给末尾添加 `.dll` 后缀, 因此只要输入动态库名称即可
* 对于 Linux, 则需要精确的输入具体的动态库名称, e.g. `libc.so.6`, 因此不适用于 成员元素 的方式来获取动态库
  * 通过 成员函数 的方式来访问 `.LoadLibrary(动态库名称)`
  * 通过动态库类的构造函数来重新定义实例

```py
cdll.LoadLibrary("libc.so.6")  
# <CDLL 'libc.so.6', handle ... at ...>
libc = CDLL("libc.so.6")       
libc
# <CDLL 'libc.so.6', handle ... at ...>  
```

### 7.1.2. 获取动态库函数  


### 7.1.3. 调用动态库函数  

## 7.2. 文档

### 7.2.1. 动态库查找

`ctypes.util.find_library(name)`:
* `name` : 动态库的名称, 不需要带任何前缀 `lib` 和后缀 `.so` 以及版本 `.6` 
* return : pathname, 如果在当前平台找不到该库则返回 `None`
* 用和 C 编译器相同的方法去查找并定位一个 动态库
* can help to determine the library to load.

