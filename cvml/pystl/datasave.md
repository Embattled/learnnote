# Data Persistence

磁盘上以持久形式存储 Python 数据

1. pickle 和 marshal 模块可以将许多 Python 数据类型转换为字节流，然后从字节中重新创建对象  
2. marshal 是 pickle 的原始模块  
3. 各种与 DBM 相关的模块支持一系列基于散列的文件格式，这些格式存储字符串到其他字符串的映射。

# pickle

实现了对一个 Python 对象结构的二进制序列化和反序列化
* "Pickling" 是将 Python 对象及其所拥有的层次结构转化为一个字节流的过程
* "unpickling" 是相反的操作
* pickle 同时也指代了同名的 protocol

Pickling（和 unpickling）也被称为`序列化`, `编组` , 或者 `平面化`。  
Pickling (and unpickling) is alternatively known as `serialization`, `marshalling,`  or `flattening`.   

* 注意：
  * pickle并不是完全安全的模组, unpickle有可能会创建一个可以执行任何代码的数据对象
  * 不要轻易unpickle一个未知来源的数据

pickle 版本 | 支持的 python 版本 | 特点
-|-|-
0| all | human-readable
1| all| binary
2| 2.3 | 支持 new-style classes
3 | 3.0| 支持 bytes, 不支持 python2, python 3.0~3.7 的默认
4 | 3.4| 优化大规模对象, 3.8~的默认
5 | 3.8 | 支持 out-of-band data

## 和其他打包模组的比较

1. python 专用, 通常不能被别的语言 unpickle
2. 可以保存并恢复数据引用和指针

* marshal 是更原始的打包方法, 目前仍然存在 stl 中主要是为了支持 `.pyc` 文件
    1. pickle 会对序列化的对象进行追踪, 如果同一组数据有多个引用, 那么不会被重复序列化
    2. 因此pickle能够处理递归对象, 而marshal不可以
    3. marshal 不能处理用户定义的类
    4. marshal 序列化的数据对象没有跨python版本的兼容性

* JSON 是字符化的打包方法, 用`utf-8`编码, 而 pickle 是直接二进制序列化
    1. JSON 人类可读
    2. JSON 是独立的数据格式, pickle 是python 专用
    3. JSON 能承载的数据基本上只有python build-in type的程度, 不能表示用户自定义类
    4. JSON 读取一个数据不会有任意代码被执行的脆弱性


## 常量

函数中的 protocol 传入一个数字参数指定协议版本, 或者传入以下的常量  
1. `pickle.HIGHEST_PROTOCOL`  替代具体版本数字, 使用 python 解释器支持的最高版本
2. `pickle.DEFAULT_PROTOCOL`  作为模组里函数参数的默认值, 根据python版本而不同


## 类 

pickle 模组定义了三个类  
* Pickler
* Unpickler
* PicklerBuffer

通用参数:  
* file : 一个文件对象, 必须有 `write()`方法接受一个single bytes, 可以是二进制打开的一个文件, 或者继承于`io.BytesIO`的任意对象
* fix_imports : 如果协议是 protocol 0,1,2 提供 python 2,3的通用兼容性, True 就完事儿了
* buffer_callback : `protocol 5` 相关的, 看不懂

### Pickler

class `pickle.Pickler(file, protocol=None, *, fix_imports=True, buffer_callback=None)`  

方法:
* `.dump(obj)`
  * 写入 file
* `.persistent_id(obj)`
  * 


## 函数

包中提供了可直接使用的函数, 可类对象相比, 这些函数主要用于快速使用  
参数的意思也完全一样  

`pickle.dump(obj, file, protocol=None, *, fix_imports=True, buffer_callback=None)`  
`pickle.dumps(obj, protocol=None, *, fix_imports=True, buffer_callback=None)`  

`pickle.load(file, *, fix_imports=True, encoding="ASCII", errors="strict", buffers=None)`  
`pickle.loads(data, /, *, fix_imports=True, encoding="ASCII", errors="strict", buffers=None)`  

## 使用方法