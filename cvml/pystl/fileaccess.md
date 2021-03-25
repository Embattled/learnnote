# 1. File and Directory Access

用于处理文件路径  

# 2. os.path

`os.path` 是一整个模块名, 从 `os` 模组中被分离出来
* 提供了一些操作路径字符串的方法
* 还包含一些或者指定文件属性的一些方法

该模块是较为低级的模块, pathlib里拥有更高级的函数及对象  
The pathlib module offers high-level path objects.  

## 2.1. 判断函数

* os.path.isdir(path) 	判断路径是否为目录。
* os.path.isfile(path) 	判断路径是否为文件。


## 2.2. 提取及转换函数

* os.path.realpath(path) 	返回 path 的真实路径。
* os.path.dirname(path) 	返回 path 路径中的目录部分。


# 3. pathlib

* The pathlib module was introduced in Python 3.4 .
包含了一些类, 操作对象是各种操作系统中使用的路径  

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
| parent   | 返回当前路径的上一级路径，相当于 `parents[0]` 的返回值。                     |
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



# shutil - High-level file operations

