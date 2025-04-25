# 1. Command Line Interface Libraries

3.13 文档重新分类出来的模组库

The modules described in this chapter assist with implementing command line and terminal interfaces for applications.

* argparse — Parser for command-line options, arguments and subcommands
* optparse — Parser for command line options
* getpass — Portable password input
* fileinput — Iterate over lines from multiple input streams
* curses — Terminal handling for character-cell displays
* curses.textpad — Text input widget for curses programs
* curses.ascii — Utilities for ASCII characters
* curses.panel — A panel stack extension for curses

https://docs.python.org/3/library/cmdlinelibs.html


# 2. argparse

argparse组件可以很方便的写一个命令行界面, 可以很容易的定义程序的参数, 并从`sys.argv`中提取出来, 同时还会自动提供错误信息  


```py
import argparse
# 建立命令行翻译器, 可以同时设置该程序的主要目的
parser = argparse.ArgumentParser(description="calculate X to the power of Y")  
args = parser.parse_args() # 翻译传入的命令行参数
```




## 2.1. argparse.ArgumentParser Objects

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


## 2.2. add_argument() - 添加一个命令

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


### 2.2.1. name or flags

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


### 2.2.2. action - API 提供的参数行为

* `store`   : 存储对应参数数值, 默认行为
* `store_const` : 参数列表中如果出现了对应 option, 则会把对应 value 设置为函数调用参数中的 `const` 值
* `store_true` /  `store_false` : 特殊种类的 `store_const`, 会将对应参数分别存储为 True, 或者 False. 对应的, 会将该参数设置默认值为 False/True  
* `append`
* `append_const`
* `extend`
* `count` : 用于技术对应的参数出现的次数, 常用来指定 verbosity levels
  * `parser.add_argument('--verbose', '-v', action='count', default=0)`
  * 
* `help`
* `version` : 特殊 action, 用于打印程序版本
  * 需要在 add_argument 的时候指定 `version` 参数
  * `parser.add_argument('--version', action='version', version='%(prog)s 2.0')`

### 2.2.3. nargs

默认情况下, 单个命令行参数是与一个动作绑定的, 但是 nargs 定义可以将多个命令行参数绑定为同一个动作  

`nargs`支持的参数:
* `N` (an integer)    : 接下来的 N 个命令行参数会被打包成 list 作为一个参数值
  * 要注意当 N=1 的时候, 仍然会创建一个包含唯一参数的 list, 这与默认情况下仍然是不同的
* `'?'` : ? 字符    :  不是很好理解 `[TODO]`
  * 
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

### 2.2.4. type

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

### 2.2.5. action - Action class

Action classes implement the Action API
* a callable which returns a callable which processes arguments from the command-line.
* Any object which follows this API may be passed as the action parameter to add_argument().




## 2.3. The parse_args() method - 解析 CLI 命令

* `parser.parse_args()` 一般就是直接无参数调用, 会直接翻译传入的命令行参数, 返回一个 `Namespace` object

### 2.3.1. Namespace - 存储命令解析结果

* `Namespace` 就是定义在 argparse 包中的一个简单的类, 和字典类似, 但是 print()更加可读 
* 可以使用 `vars()` 方法进行转换成字典
* `args.*` 用点号进行对应的参数访问

通过在 parse_args 中添加 namespace 参数可以在某个已经存在的基础上再次进行 CLI 解析
`parser.parse_args(args=['--foo', 'BAR'], namespace=c)`


## 2.4. 高级 args

除了基础的直接对 parser 里添加参数外, 还有其他特殊的参数类别  

* ArgumentParser.add_argument_group()
* ArgumentParser.add_mutually_exclusive_group
* ArgumentParser.add_subparsers
* ArgumentParser.set_defaults

### 2.4.1. Argument groups 参数分组

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

### 2.4.2. mutual exclusion 矛盾参数

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

### 2.4.3. subparsers 子解释器

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

### 2.4.4. Parser defaults

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

## 2.5. Parser defaults


# 3. optparse — Parser for command line options

在 3.13 之前一直是 Deprecated 函数, 但是在 3.13 的时候又被加回了标准库

在某些情况下, 需要自己开发另外的 CLI 处理库 (自定义的argparse) 的情况下, 可以使用 optparse 进行更底层的自定义

# 4. getpass  — Portable password input 

类似于 argparse , 只是该模组只针对密码界面    

不可以在 WASI 系统下使用

The getpass module provides two functions:
* `getpass.getpass(prompt='Password: ', stream=None)`
  * prompt : 提示信息
  * 可以用字符串存储得到的密码 `input_str=getpass.getpass()`
  * 这个界面会保证密码不会显式在屏幕上
  * 调用该函数会立即进去密码输入界面  
* `getpass.getuser()`
  * 获取当前进程的用户名
  * This function checks the environment variables `LOGNAME`, `USER`, `LNAME` and `USERNAME`, in order, and returns the value of the first one which is set to a `non-empty string`. 
  * In general, this function should be preferred over `os.getlogin()`
    * 该函数旨在替代 os.getlogin()
