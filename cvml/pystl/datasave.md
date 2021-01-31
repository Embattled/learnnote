# Data Persistence

磁盘上以持久形式存储 Python 数据

1. pickle 和 marshal 模块可以将许多 Python 数据类型转换为字节流，然后从字节中重新创建对象  
2. marshal 是 pickle 的原始模块  
3. 各种与 DBM 相关的模块支持一系列基于散列的文件格式，这些格式存储字符串到其他字符串的映射。

# pickle

实现了对一个 Python 对象结构的二进制序列化和反序列化
* "Pickling" 是将 Python 对象及其所拥有的层次结构转化为一个字节流的过程
* "unpickling" 是相反的操作


Pickling（和 unpickling）也被称为`序列化`, `编组` , 或者 `平面化`。  
Pickling (and unpickling) is alternatively known as `serialization`, `marshalling,`  or `flattening`.   



