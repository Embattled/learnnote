# 1. File and Directory Access

用于处理磁盘上的文件以及目录

* pathlib — Object-oriented filesystem paths

## 1.1. path-like object

一类对象, 用于代表 python 路径相关的 常见的可能参数
* `str` or `bytes` object representing a path
*  an object implementing the `os.PathLike` protocol
*  实现了 `os.PathLike` 的对象可以通过 os 下的 `os.fspath()` 转换成 str or bytes


# 2. os.path

`os.path` 是一整个模块名, 从 `os` 模组中被分离出来
* 提供了一些操作路径字符串的方法
* 还包含一些或者指定文件属性的一些方法
* 该包的所有函数都接受  only bytes or only string objects as their parameters.

该模块是较为低级的模块, pathlib里拥有更高级的函数及对象  
The pathlib module offers high-level path objects.  
* 但是 os.path 用的非常多  
* 内容非常简单

## 2.1. is 判断函数

* os.path.isdir(path) 	判断路径是否为目录。
* os.path.isfile(path) 	判断路径是否为文件。
* os.path.isabs(path)   判断是否是一个绝对路径
  * linux 下即 `/` 开头
  * windows下即 盘符: 加反斜杠开头

## 2.2. 路径检测

* os.path.exists(path)
  * 判断该路径是否存在
  * 如果是 破坏掉的 link , 则返回 False
* os.path.lexists(path)
  * 对于 破坏掉的 link 也返回 True
  * 只检测是否存在该文件

## 2.3. 提取及转换函数

一分为二函数  
* `os.path.basename(path)`  返回 path 路径中的最后一个成分
  * 如果 path 是文件路径, 就是文件名
  * 如果 path 是目录, 就是叶子文件夹名
* `os.path.dirname(path)` 	返回 path 路径中的 减去 basename(path)
  * 通过 `os.path.realpath(__file__)` 来提取脚本所在目录
* `os.path.split(path)`     将路径分解成 head 和 tail 两部分, 分别相当于上面的函数
  * join(head, tail) returns a path to the same location as path

* `os.path.realpath(path)` 	返回 path 的真实路径。
* `os.path.abspath(path)`   返回相对路径 path 的绝对路径版本
  * 相当于 `os.path.normpath(os.path.join(os.getcwd(), path))`
  * 直接执行 `os.path.abspath('.')` 获取当前路径

## 2.4. 路径操作

* os.path.join(paths) 该包最重要的函数, 自动生成路径, 不用考虑 `/` 的问题
  * 接受一系列的参数
  * 每个参数作为一个文件名目录
  * 最后一个参数后面不接 `/`
  * 如果某个元素是绝对路径, 那么该元素之前的输入会被抛弃 (windows下是 `c:` 之类的)
* os.path.normpath(path) 标准化一个路径
  * 删除多余的 `/`
  * 删除最末尾的 `/`
  * 删除位于中间的 `..` 并进行逻辑重定位


# 3. pathlib — Object-oriented filesystem paths

* 尽管 pathlib 实现了 str() 函数, 但该路径对象对其他非标准库的兼容性并不好

* The pathlib module was introduced in Python 3.4 .
* 包含了一些类, 操作对象是各种操作系统中使用的路径  
* 相比于os.path, pathlib 将路径进行了对象化, 并提供了遍历文件夹相关的操作


pathlib 实现了 Unix 和 Windows 两种不同格式的路径类, 类的具体继承定义如下, 明确各个类的负责的功能对于使用帮助很大  
1. PurePath   : 定义了一个路径类的各种成员, 作为该包的基类, 用于明确 Path 类的成员变量构成
2. 1 PurePosixPath : 定义了 Unix 路径的规格
3. 2 PureWindowsPath : 同理, Windows 路径
4. Path(PurePath)  : 添加了具体的OS访问方法, 是主要接口  
5. 1 PosixPath(Path,PurePosixPath) : 具体的 Unix 路径类
6. 2 WindowsPath(Path,PureWindowsPath) : 具体的 Windows 路径类

在定义一个具体的路径对象时, 直接使用 Path()即可, 会根据系统标识号自动选择对应的对象  
要想操作不同系列的路径, 即 Unix 操作 Windows 路径, 不能够定义 WindowsPath 类, 因为其中实现了 OS 接口, 但是可以定义 PureWindowsPath, 因为其中只有路径的字符操作没有 OS 操作  


## 3.1. pathlib.PurePath

pathlib模块中的基类, 将路径看作普通的字符串
* 将多个指定的字符串拼接成适用于当前操作系统的路径格式
* 判断两个路径是否相等
* 该模块的意义是 没有OS接口方法

PurePath作为该模块的基类, 提供了最基础的构造方法和实例属性
1. 要创建Pure路径时, 直接创建 PurePath 对象即可, 解释器会自动根据操作系统返回 PurePosixPath或者 PureWindowsPath
2. 创建好后可以通过 str() 转换成字符串

要注意, Unix 和 Windows 里：
* 路径对于大小写的敏感是不同的, Unix 是大小写敏感的, windows 不是, 这一点在 PurePath 里也是实现了的
* 子路径的符号是不同的, 在 Python 里可以统一的使用斜杠/ , 在输出的时候会自动转换成对应操作系统的分隔符  

The best way to construct a path is to join the parts of the path using the special operator `/`.


### 3.1.1. 创建路径

* 在创建对象时,传入多个字符串, 自动生成对应系统的路径  
* 如果不传入参数, 等同于只传入 `'.'`   表示当前路径
* 构造函数具有鲁棒性, 多余的点或者斜杠会被忽略

```py
path = PurePath('http:','c.biancheng.net','python')
# http:\c.biancheng.net\python  windows 下
# http:/c.biancheng.net/python  linux 下
print(path)
```

### 3.1.2. PurePath 的运算符重载  / 

PurePath 类实现了 / 运算符, 直接使用斜杠可以快速的为 Path 添加子路径  

```py

a = PurePath('test')
a / "cd"
>>> PurePath('/test/cd')

```

### 3.1.3. 提取路径成分

全部都是`PurePath.` 的成员变量, 直接用即可

| 方法名   | 功能                                                                       |
| -------- | -------------------------------------------------------------------------- |
| parts    | 返回路径字符串中所包含的各部分。                                           |
| drive    | 返回路径字符串中的驱动器盘符。                                             |
| root     | 返回路径字符串中的根路径。                                                 |
| anchor   | 返回路径字符串中的盘符和根路径。                                           |
| parents  | 特殊 object, 当前路径的全部父路径, len(parents) 相当于目录个数             |
| parent   | 返回当前路径的上一级路径，相当于 `parents[0]` 的返回值。                   |
| name     | 返回当前路径中的文件名, 相当于剔除 parent                                  |
| suffixes | 返回当前路径中的文件所有后缀名。                                           |
| suffix   | 返回当前路径中的文件后缀名。相当于 suffixes 属性返回的列表的最后一个元素。 |
| stem     | 返回当前路径中的主文件名, 剔除掉 suffix 的 name                            |

```py

path1 = Path('.') / 'folder1' / 'text1.txt'
print([path1, path1.name, path1.stem, path1.suffix, path1.parent, path1.parent.parent, path1.anchor])
# [PosixPath('folder1/text1.txt'), 'text1.txt', 'text1', '.txt', PosixPath('folder1'), PosixPath('.'), '']
```
## 3.2. pathlib.Path - Concrete paths

* Path类是PurePath的子类, 因此继承的方法不多赘述
* Path类的路径必须是真实有效的

提供的方法 : 各种 OS 路径访问  
* 判断路径是否真实存在
* 判断该路径对应的是文件还是文件夹
* 如果是文件，还支持对文件进行读写等操作

很多都是从 os. 包中进行再封装的函数  

### 3.2.1. 创建路径

类构造函数创建
* `pathlib.Path(*pathsegments)`
  * `path1 = Path('.') / 'folder1' / 'text1.txt'`
  * 会根据操作系统自动创建对应的 Path, 因此一般不需要用到下面的函数
  * `pathlib.PosixPath(*pathsegments)`
  * `class pathlib.WindowsPath(*pathsegments)`


便捷类方法创建
* 创建当前路径的 Path
  * `Path.cwd()`
  * `Path('.')`
* 创建用户主目录 Path
  * `Path.home()`

路径的解释与转换
* 解释 用户主目录符号 `~`
  * `Path('~').expanduser()  ->  PosixPath('/home/longubuntu') `
  * 注意, 带有 `~` 的 path 必须在通过该解释函数后才能正确使用
* 转化成绝对路径
  * `Path.resolve(strict=False)`
  * 解释路径中的所有 `.`  `..` 等
  * `strict=True` 相当于同时执行了 Path.exists(), 属于是一个懒人参数
* 转化(追踪) link
  * `Path.readlink()` 追踪一个link 的 path, 即返回 path of target file 


### 3.2.2. 获取文件信息

简单函数 省略 Path.*

| 名称                          | 功能                                                                                    |
| ----------------------------- | --------------------------------------------------------------------------------------- |
| owner()                       | Return the name of the user owning the file                                             |
| group()                       | Return name of the group owning the file                                                |
| stat(*, follow_symlinks=True) | 返回一个 `os.stat_result` 对象, 具体查看该对象的成员, 传入 False 则查看 link 本身的信息 |
| lstat()                       | 懒人函数, 相当于 stat(follow_symlinks=False)                                            |


### 3.2.3. 更改文件信息


`Path.chmod(mode, *, follow_symlinks=True)`
* Change the file mode and permissions, like `os.chmod()`
* e.g. `p.chmod(0o444)`
* `Path.lchmod(mode)`  懒人函数, 相当于 `chmod(mode, follow_symlinks=False)`

`Path.rename(target)`
* 重命名/移动一个 文件(夹)
* 如果 target 已经存在
  * Unix : 会静默覆盖目标文件
  * Windows : 会报错

`Path.replace(target)`
* 重命名/移动一个 文件(夹)
* rename 的确保覆盖版本, 目标除了 file 以外, 还可以是 empty folder


### 3.2.4. 文件(夹) 创建与访问

文件夹 增删
* `Path.mkdir(mode=0o777, parents=False, exist_ok=False)`
  * Path 的创建文件夹, 比较方便
`Path.rmdir()`
  * 删除一个文件夹, 该文件夹必须为空  

`Path.open(mode='r', buffering=- 1, encoding=None, errors=None, newline=None)`
* 返回一个 opened file
* 相当于 `open(Path)`


文件读写函数, 直接省去 .open() 的过程
* `Path.read_bytes()` 文件内容以 byte 读取, 返回 `b'` 二进制内容
* `Path.read_text(encoding=None, errors=None)` : 文件内容以 txt 读取




### link 的创建与修改

链接的创建以及 Follow
* `Path.readlink()` 追踪一个link 的 path, 即返回 path of target file 
* `Path.symlink_to(target, target_is_directory=False)`

### 3.2.5. is_* 系列判断函数

该部分统一返回 bool, 省略 Path.*

总是返回 False 的特殊情况
* 不存在的路径
* broken symlink
* Permission error 无访问权限之类的


| 函数              | 功能                                               |
| ----------------- | -------------------------------------------------- |
| exists()          | 路径是否存在, 如果不存在则以下所有函数都是 False   |
| is_dir()          | 文件夹, 或者 symbolic link pointing to a directory |
| is_file()         | 文件,  symbolic link pointing to a regular file    |
| is_mount()        | Unix 专属函数, 不太懂                              |
| is_symlink()      | True if the path points to a symbolic link         |
| is_socket()       | True if the path points to a Unix socket           |
| is_fifo()         | True if the path points to a FIFO                  |
| is_block_device() | True if the path points to a block device          |
| is_char_device()  | True if the path points to a character device      |


### 3.2.6. 检索文件夹


* `.iterdir()` 获取一层文件列表, 注意该函数返回的是一个迭代器, 需要手动转化成list * 

```py
path2 = Path('.') / 'folder1'
path_list = list(path2.iterdir())
print(f'Number of files: {len(path_list)}')
```

* `Path.glob(pattern)`  文件 pattern 检索  : 返回的也是迭代器
  * 使用 `*` 通配符来匹配所有文件名   `glob(*.py)`
  * 使用 `**` 来递归的检索所有子文件夹 `glob(**/*.py)`
* `Path.rglob(pattern)` 递归的 pattern 检索
  * 懒人函数
  * 相当于默认在 pattern 前面添加了 `**/`

### 3.2.7. 与 OS. 的互换性

由于很多函数都是 os. 的封装, 文档中列出了与 os 的函数对照表

https://docs.python.org/3/library/pathlib.html?highlight=pathlib#correspondence-to-tools-in-the-os-module 


# 4. shutil - High-level file operations


# 5. 简单功能模组

* filecmp   : 面向路径的字符串比较
* fnmatch   : 通配符匹配路径
* glob      : 同样的是查找路径. 比 fnmatch 高级

* tempfile  : 方便的创建临时文件

## 5.1. fnmatch - Unix filename pattern matching

该模组支持 Unix-shell 的通配符: `* ? [seq] [!seq]`, 但不支持完整正则表达式
* 作为 glob 的底层函数, 用于验证一个路径是否满足 pattern

`fnmatch.fnmatch(filename, pattern)`
* Test whether the filename string matches the pattern string.
* Returning `True` or `False`.
* 该比较是大小写不敏感的 (内部通过对两个参数应用 `os.path.normcase()` 来实现)

`fnmatch.fnmatchcase(filename, pattern)->bool`
* 大小写敏感的比较

```py
import fnmatch
import os

for file in os.listdir('.'):
    if fnmatch.fnmatch(file, '*.txt'):
        print(file)
```

`fnmatch.filter(names, pattern)`
* names 是一个可迭代的对象, 过滤掉不满足 pattern 的元素, 返回新的 `list`
* 相当于 `[n for n in names if fnmatch(n, pattern)]` 


`fnmatch.translate(pattern)`
* 因为 shell 的三类通配符和 正则表达式的语法规则不一致
* 该函数用于将 shell-stype 的 pattern 转化成对应的 正则表达式, 可以用于 `re.match()` 中
* `*.txt` -> `'(?s:.*\\.txt)\\Z'` (这块复制的没看懂, 可能是 windows 下的正则路径)

## 5.2. glob - Unix style pathname pattern expansion


glob 模组提供查找对应 pattern 的路径的功能， 该模组顺从 Unix shell 的规则
* results are returned in arbitrary order
* 可以使用 `* ? []`三类通配符 (wildcards)
* 底层实现是通过 `os.scandir()` 和 `fnmatch.fnmatch()` 来实现的, 并不是真的调用一个 subshell
* 相比于 `fnmatch.fnmatch()`, glob 会把 `.` 开头的文件识别为 特殊 cases


`glob.glob(pathname, *, root_dir=None, dir_fd=None, recursive=False)`
* 返回满足 pathname 的路径 list, 可能会返回空 list
* 注意 broken symlinks 也会被返回
  * `pathname`  : 必须是str, 可以是绝对 or 相对路径, 可以使用三类通配符
  * `root_dir`  : None, or path-like, 指定要搜索的目录根路径, 相当于在执行 glob 之前使用 cd, 同理只在 pathname 为相对路径的时候有作用
  * `recursive` : True 的时候, the pattern “**” will match any files and zero or more directories, subdirectories and symbolic links to directories.



`glob.iglob(pathname, *, root_dir=None, dir_fd=None, recursive=False)`
* 功能和参数完全一致
* 返回值为 `iterator`, 即不会同时存储所有值在内存中



`glob.escape(pathname)`
* 用于服务 glob 的函数
* 直接转义 pathname 中的所有特殊符号 `*?[]`
* 对于想查找可能包含特殊符号的文件, 但又不知道怎么写转义符的时候很有用

## 5.3. tmpfile

只负责临时文件的全部, 分为 
* low-level : which require manual cleanup. 需要手动删除使用完毕的 temp 文件
  * mkstemp()
  * mkdtemp()
* high-level: which provide automatic cleanup and can be used as context managers
  * TemporaryFile
  * NamedTemporaryFile
  * TemporaryDirectory
  * SpooledTemporaryFile

### 5.3.1. 部件命令

创建随机文件名
* gettempprefix()   : Return the filename prefix used to create temporary files
* gettempprefixb()  : return value is in bytes.

### 5.3.2. 基础命令

* mkstemp : Creates a temporary file in the most secure manner possible.
* mkdtemp : Creates a temporary directory in the most secure manner possible.

```py
tempfile.mkstemp(suffix=None, prefix=None, dir=None, text=False)
# returns a tuple containing an OS-level handle to an open file (as would be returned by os.open()) 
# and the absolute pathname of that file
# in that order.
tempfile.mkdtemp(suffix=None, prefix=None, dir=None)
# returns the absolute pathname of the new directory.

# 参数
# suffix  : 文件(夹)的后缀名, 如果是文件类型, 需要手动加入 `.`
# prefix  : 文件(夹)的后缀名, 默认不为空, 而是  gettempprefix() or gettempprefixb() 的结果
# dir     : 文件(夹)的路径, 有 dir 则会在该路径创建, 否则会检查环境变量 TMPDIR, TEMP or TMP, 最后是根据系统来确定 tmp 路径
# text    : 该临时文件是否被文本模式打开, 如果 false 则是二进制模式
```