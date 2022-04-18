# 1. File and Directory Access

用于处理文件路径  

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
* os.path.basename(path)  返回 path 路径中的最后一个成分
  * 如果 path 是文件路径, 就是文件名
  * 如果 path 是目录, 就是叶子文件夹名
* os.path.dirname(path) 	返回 path 路径中的 减去 basename(path)
* os.path.split(path)     将路径分解成 head 和 tail 两部分, 分别相当于上面的函数
  * join(head, tail) returns a path to the same location as path

* os.path.realpath(path) 	返回 path 的真实路径。
* os.path.abspath(path)   返回相对路径 path 的绝对路径版本
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


# 3. pathlib

* The pathlib module was introduced in Python 3.4 .
* 包含了一些类, 操作对象是各种操作系统中使用的路径  
* 相比于os.path, pathlib 将路径进行了对象化, 并提供了遍历文件夹相关的操作

## 3.1. pathlib.PurePath

pathlib模块中的基类, 将路径看作普通的字符串
* 将多个指定的字符串拼接成适用于当前操作系统的路径格式
* 判断两个路径是否相等

PurePath作为该模块的基类, 提供了最基础的构造方法和实例属性
1. 创建路径时, 直接创建 PurePath 对象即可, 解释器会自动根据操作系统返回 PurePosixPath或者 PureWindowsPath
2. 创建好后可以通过 str() 转换成字符串

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

### 3.1.2. 提取路径成分

全部都是`PurePath.` 的成员变量, 直接用即可

| 方法名   | 功能                                                                       |
| -------- | -------------------------------------------------------------------------- |
| parts    | 返回路径字符串中所包含的各部分。                                           |
| drive    | 返回路径字符串中的驱动器盘符。                                             |
| root     | 返回路径字符串中的根路径。                                                 |
| anchor   | 返回路径字符串中的盘符和根路径。                                           |
| parents  | 返回当前路径的全部父路径。                                                 |
| parent   | 返回当前路径的上一级路径，相当于 `parents[0]` 的返回值。                   |
| name     | 返回当前路径中的文件名。                                                   |
| suffixes | 返回当前路径中的文件所有后缀名。                                           |
| suffix   | 返回当前路径中的文件后缀名。相当于 suffixes 属性返回的列表的最后一个元素。 |
| stem     | 返回当前路径中的主文件名。                                                 |

```py

path1 = Path('.') / 'folder1' / 'text1.txt'
print([path1, path1.name, path1.stem, path1.suffix, path1.parent, path1.parent.parent, path1.anchor])
# [PosixPath('folder1/text1.txt'), 'text1.txt', 'text1', '.txt', PosixPath('folder1'), PosixPath('.'), '']
```
## 3.2. pathlib.Path

* Path类是PurePath的子类, 因此继承的方法不多赘述
* Path类的路径必须是真实有效的

提供的方法 :
* 判断路径是否真实存在
* 判断该路径对应的是文件还是文件夹
* 如果是文件，还支持对文件进行读写等操作

The best way to construct a path is to join the parts of the path using the special operator `/`.


### 3.2.1. 定义路径

You can use `Path.cwd()` or `Path('.') `to refer to your currently working directory.
```py
from pathlib import Path

print("Getting 'text1.txt'")
path1 = Path('.') / 'folder1' / 'text1.txt'
print(path1)

```

### 3.2.2. .iterdir() 获取文件列表

Using `.iterdir()` you can get all the files in a folder.   
By list comprehension, you can convert this into a list object.  

```py
path2 = Path('.') / 'folder1'
path_list = list(path2.iterdir())
print(f'List of files: {path_list}')
'''
List of files: [PosixPath('folder1/text1.txt'), PosixPath('folder1/text2.txt'), PosixPath('folder1/text3.txt')]
'''

print(f'Number of files: {len(path_list)}')

```

### 3.2.3. Path 的有用方法

```py

# Path.exists()
# Checks if a path exists or not. Returns boolean value.
file_path = Path('.') / 'folder1' / 'text2.txt'
print(file_path.exists())


# Path.glob()   Globs and yields all file paths matching a specific pattern. 
#  mark (?), which stands for one character.
print("\nGetting all files with .csv extension.")
dir_path = Path('.') / 'folder1'
file_paths = dir_path.glob("*.csv")
print(list(file_paths))

# Path.rglob()
# This is like Path.glob method but matches the file pattern recursively.
print("\nGetting all .txt files starts with 'other' in all directories.")
dir_path = Path('.')
file_paths = dir_path.rglob("other?.txt")
print(list(file_paths))

# Path.mkdir()
# Creates a new directory at this given path. 
dir_path = Path('.') / 'folder_new' / 'folder_new_1'
# parents:(boolean) If parents is True, 
#         any missing parents of this path are created as needed. 
#          Otherwise, if the parent is absent, FileNotFoundError is raised.
dir_path.mkdir(parents=True)
# exist_ok: (boolean) 
#       If False, FileExistsError is raised if the target directory already exists. 
#       If True, FileExistsError is ignored.

 
# Path.rename(target)   This will raise FileNotFoundError if the file is not found
dir_path = Path('.') / 'folder_new' / 'folder_new_1'
dir_path.rename(dir_path.parent / 'folder_n1')


# Replaces a file or directory to the given target. Returns the new path instance. 
dir_path = Path('.') / 'folder_new' / 'folder_n1'
dir_path2 = Path('.') / 'folder1'  
dir_path.replace(dir_path.parent / dir_path2)



# Path.rmdir()
# Removes a path pointing to a file or directory. The directory must be empty, otherwise, OSError is raised.
```
## 3.3. PosixPath WindowsPath

* PurePosixPath PureWindowsPath 继承自PurePath
* PosixPath WindowsPath 各自继承Pure*和Path类
作为实例化的类, 一般不需要手动定义, 解释器会自动根据系统将Path和PurePath实例化成对应的类



# 4. shutil - High-level file operations


# 5. 简单功能模组

* fnmatch
* glob

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

