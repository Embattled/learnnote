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
* 
##

### os.walk

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

## 3.2. 相关定义

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

## 3.1. 获取时间

无参数函数
1. time()    : 获取浮点数表示的从 epoch 开始经过的秒数

## 获取规格化字符串时间

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

# logging 日志模块

