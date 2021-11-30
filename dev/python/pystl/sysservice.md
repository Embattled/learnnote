# 1. Generic Operating System Services

* 和 runtime 中的 python 解释器模块不同, 这部分的模块更贴近操作系统方面
* 在几乎所有的操作系统都能使用, 提供了与操作系统相关的所有操作的接口  

# 2. os

* os — Miscellaneous operating system interfaces, 包含了各种操作系统相关的方法
* 一些非常常用的功能被移动到了python内部函数:
  * 如果想使用文件IO, 查看内部函数 `open()`即可, 它是 `os.fdopen()` 的alias
  * 管理运行时 paths, 使用模组 `os.path`
  * 文件读取的相关内容在 `fileinput` 中
  * 临时文件 `tempfile`
  * 高级路径文件操作 `shutil`


* 非常大的库
  

## Files and Directories

os.mkdir(path, mode=0o777, *, dir_fd=None)

os.makedirs(name, mode=0o777, exist_ok=False)

### 2.0.1. os.walk

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

# 3. time

* 与系统时间相关的函数, 大部分都是直接调取操作系统内置的 同名C函数  
* Python 中与时间操作的 module 还有 `datetime` 和 `calendar`

并不是所有的函数都可以在所有平台使用, 因为大部分函数都是直接调用 C 函数库

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

无参数函数
1. time()    : 获取浮点数表示的从 epoch 开始经过的秒数

## 3.3. 获取规格化字符串时间

`time.strftime(format, t=None )`
* 将一个元组或者 struct_time 转化成格式的字符串
  * t 可以接受 gmtime() 或者 localtime() 的输入
  * 没有 t 的话代表输出当前的时间

| 描述符 | 意义                  |
| ------ | --------------------- |
| %S     | 秒数                  |
| %M     | 分钟数                |
| %H %I  | 24-小时 12-小时       |
| %w     | 星期几 0表星期天      |
| %d     | 日期                  |
| %m     | 月份                  |
| %Y %y  | 四位数年份 两位数年份 |

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

## 5.1. add_argument()

* 基础上使用 `parser.add_argument()` 来添加一个命令行参数  
* 添加可选参数的时候使用 `-` 或者 `--`, 会自动识别, 且在调用值的时候不需要加 `-`
  * 同时有 `-` 和 `--` 的时候会选择 `--` 的名字作为参数调用名

```py
ArgumentParser.add_argument(
  name or flags...
  [, action]
  [, nargs]
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


**argparse的使用**
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

### 5.1.1. Action class


## 5.2. 高级 args

### 5.2.1. mutual exclusion 矛盾参数

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

### 5.2.2. Argument groups 参数分组

* 可以将不同的参数归到一组里,
* 这个组在参数调用的时候没有任何卵用
* 但是可以在程序帮助信息打印的时候更加清晰明确


```py
parser = argparse.ArgumentParser()
group = argparse.add_argument_group(title=None, description=None)
# 可以设置组的名称和说明 

group.add_argument('--foo', action='store_true')
group.add_argument('--bar', action='store_false')
# 添加至这个组的参数在打印参数说明的时候会单独分隔开, 方便区分
```

## 5.3. Print help

