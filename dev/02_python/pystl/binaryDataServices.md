# Binary Data Services

some basic services operations for manipulation of binary data. 

放置了一些基础的二进制数据操作方法, 对于有具体目的的二进制对象 (e.g. file, network protocols) 放置在其他的章节
除此之外, 一些 Text Processing Services 的库与该章节有功能上的重合, 例如 `re`  `difflib`

Python Built-in 的类 `bytes` `bytearray` `memoryview` 具有类似的简化的功能


# struct — Interpret bytes as packed binary data

提供了一个最全面的, 用于 C 结构和 Python 数据结构的二进制转换方法.  

对应的 `format strings` 描述了相对简单的版本   

该模块的函数可以用于两个目的
* 与外部元 (文件或者网络连接) 的数据交换
* Python 应用程序与 C Layer 之间的数据交换

大小端以及数据格式的字节数等信息默认参照 host, 如果涉及到网络上的交换则需要 程序员自己选定对应的解析模式  

struct 的缓冲区设计遵循了 python 的 Buffer Protocol, 因此可以方便的与其他字节流类型交互信息  

## Format Strings

在 struct 中, 用于描述一个字节流数据的格式的字符串. 包括了
* format characters     : 描述数据类型
* special characters 特殊的字符用于指定
  * byte order
  * size
  * alignment
* optional prefix character:
  * 数据的整体属性  overall properties of the data
  * actual data values 
  * padding

## Function


`struct.*` 函数介绍
* `unpack(format, buffer)`   : 从一个 buffer 解包数据,  即使数据只有一个也会返回一个 `tuple`.  buffer 的 size 必须与 format 所需要的大小相等


## class struct.Struct(format)