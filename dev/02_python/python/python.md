- [1. Python 的背景](#1-python-的背景)
  - [1.1. 版本区别  2和3](#11-版本区别--2和3)
  - [1.2. Python PEP文档](#12-python-pep文档)
  - [1.3. 底层语言](#13-底层语言)
- [2. Python的语法](#2-python的语法)
  - [2.1. Python 书写规范 (PEP 8)](#21-python-书写规范-pep-8)
  - [2.2. Python 保留字](#22-python-保留字)
  - [2.3. python 的类型提示](#23-python-的类型提示)
  - [2.4. 作用域](#24-作用域)
- [3. 内置函数 Build-in Function](#3-内置函数-build-in-function)
  - [3.1. 类函数](#31-类函数)
  - [3.2. 类型转换函数](#32-类型转换函数)
  - [3.3. 作用域变量获取函数](#33-作用域变量获取函数)
  - [3.4. print()](#34-print)
    - [3.4.1. 格式化输出](#341-格式化输出)
    - [3.4.2. 控制输出方式](#342-控制输出方式)
  - [3.5. input()](#35-input)
  - [3.6. open() 基础文件操作](#36-open-基础文件操作)
    - [3.6.1. read](#361-read)
    - [3.6.2. write](#362-write)
    - [3.6.3. position](#363-position)
  - [3.7. range()](#37-range)
  - [3.8. zip()](#38-zip)
  - [3.9. reserved()](#39-reserved)
  - [3.10. sorted()](#310-sorted)
  - [3.11. enumerate() 遍历对象函数](#311-enumerate-遍历对象函数)
- [4. Python的操作符与变量](#4-python的操作符与变量)
  - [4.1. 基础类型](#41-基础类型)
  - [4.2. 转义字符](#42-转义字符)
  - [4.3. 运算符](#43-运算符)
- [5. python Built-in Types](#5-python-built-in-types)
  - [5.1. 序列](#51-序列)
    - [5.1.1. 序列通用](#511-序列通用)
      - [5.1.1.1. 序列内置函数](#5111-序列内置函数)
      - [5.1.1.2. 序列的引用](#5112-序列的引用)
      - [5.1.1.3. 存在检查 in](#5113-存在检查-in)
      - [5.1.1.4. 索引](#5114-索引)
      - [5.1.1.5. 切片](#5115-切片)
      - [5.1.1.6. 序列相加](#5116-序列相加)
      - [5.1.1.7. 序列相乘](#5117-序列相乘)
    - [5.1.2. 列表类型 list](#512-列表类型-list)
      - [5.1.2.1. 列表的创建使用](#5121-列表的创建使用)
      - [5.1.2.2. 元素添加](#5122-元素添加)
      - [5.1.2.3. 元素删除](#5123-元素删除)
      - [5.1.2.4. 元素修改](#5124-元素修改)
      - [5.1.2.5. 列表的查找](#5125-列表的查找)
      - [5.1.2.6. 列表的赋值](#5126-列表的赋值)
      - [5.1.2.7. 增强的赋值操作](#5127-增强的赋值操作)
      - [5.1.2.8. 列表类型的方法](#5128-列表类型的方法)
    - [5.1.3. 元组](#513-元组)
      - [5.1.3.1. 元组创建](#5131-元组创建)
      - [5.1.3.2. 元组的访问和修改](#5132-元组的访问和修改)
    - [5.1.4. 二进制序列 bytes](#514-二进制序列-bytes)
      - [5.1.4.1. bytes](#5141-bytes)
      - [5.1.4.2. bytearray](#5142-bytearray)
  - [5.2. 字典 dict](#52-字典-dict)
    - [5.2.1. 字典创建删除](#521-字典创建删除)
    - [5.2.2. 访问使用](#522-访问使用)
    - [5.2.3. 成员函数操作](#523-成员函数操作)
  - [5.3. 集合 set](#53-集合-set)
    - [5.3.1. 创建](#531-创建)
    - [5.3.2. 访问](#532-访问)
    - [5.3.3. 元素操作](#533-元素操作)
    - [5.3.4. 集合运算](#534-集合运算)
  - [5.4. 字符串 string](#54-字符串-string)
    - [5.4.1. 字符串的输入](#541-字符串的输入)
    - [5.4.2. 字符串编码](#542-字符串编码)
      - [5.4.2.1. 大小写及内容检测方法](#5421-大小写及内容检测方法)
      - [5.4.2.2. 开头结尾检测方法](#5422-开头结尾检测方法)
      - [5.4.2.3. 组合与切割方法](#5423-组合与切割方法)
      - [5.4.2.4. 对齐方法](#5424-对齐方法)
      - [5.4.2.5. 清除对齐方法](#5425-清除对齐方法)
      - [5.4.2.6. 识别转换方法 **str.extract()**](#5426-识别转换方法-strextract)
      - [5.4.2.7. str.get\_dummies()](#5427-strget_dummies)
  - [5.5. Iterator](#55-iterator)
    - [5.5.1. Generator Types](#551-generator-types)
- [6. Python 流程控制](#6-python-流程控制)
  - [6.1. 逻辑流程](#61-逻辑流程)
    - [6.1.1. `if` 表达式](#611-if-表达式)
    - [6.1.2. while 表达式](#612-while-表达式)
    - [6.1.3. for 表达式](#613-for-表达式)
    - [6.1.4. 列表推导式（又称列表解析式）](#614-列表推导式又称列表解析式)
  - [6.2. 异常流程控制](#62-异常流程控制)
    - [6.2.1. 异常类](#621-异常类)
    - [6.2.2. raise 语句](#622-raise-语句)
    - [6.2.3. assert 语句](#623-assert-语句)
    - [6.2.4. 异常信息捕获](#624-异常信息捕获)
    - [6.2.5. 自定义异常类](#625-自定义异常类)
- [7. python 的函数](#7-python-的函数)
  - [7.1. 函数参数](#71-函数参数)
    - [7.1.1. Python的可变参数](#711-python的可变参数)
    - [7.1.2. 逆向参数收集](#712-逆向参数收集)
    - [7.1.3. partial 偏函数](#713-partial-偏函数)
    - [7.1.4. 闭包函数](#714-闭包函数)
  - [7.2. 函数的文档](#72-函数的文档)
  - [7.3. yield 表达式](#73-yield-表达式)
  - [7.4. lambda 表达式 匿名函数](#74-lambda-表达式-匿名函数)
  - [7.5. 函数的异常处理](#75-函数的异常处理)
- [8. python 的类](#8-python-的类)
  - [8.1. 定义](#81-定义)
  - [8.2. self](#82-self)
  - [8.3. 类的变量](#83-类的变量)
  - [8.4. 类方法 静态方法](#84-类方法-静态方法)
  - [8.5. 类的描述符](#85-类的描述符)
  - [8.6. 类的封装](#86-类的封装)
    - [8.6.1. property()](#861-property)
    - [8.6.2. @property 装饰器](#862-property-装饰器)
  - [8.7. 类的继承和多态](#87-类的继承和多态)
    - [8.7.1. super](#871-super)
    - [8.7.2. MRO Method Resolution Order](#872-mro-method-resolution-order)
- [9. python 的模块和包](#9-python-的模块和包)
  - [9.1. 导入模块或包](#91-导入模块或包)
  - [9.2. 自定义模块](#92-自定义模块)
  - [9.3. 包](#93-包)
  - [9.4. 包信息调取](#94-包信息调取)
- [10. Python 的文件操作](#10-python-的文件操作)
  - [10.1. open 打开文件](#101-open-打开文件)
  - [10.2. 读取文件](#102-读取文件)
  - [10.3. 写入文件](#103-写入文件)
  - [10.4. 结构化读取文件](#104-结构化读取文件)
- [11. 正则表达式 re包](#11-正则表达式-re包)
  - [11.1. 使用正则表达式的基础函数](#111-使用正则表达式的基础函数)
  - [11.2. 正则表达式-单字符](#112-正则表达式-单字符)
  - [11.3. 正则表达式-多次匹配](#113-正则表达式-多次匹配)
- [12. Python 的包 环境管理](#12-python-的包-环境管理)
  - [12.1. Python的包管理](#121-python的包管理)
    - [12.1.1. distutils 和 setuptools](#1211-distutils-和-setuptools)
  - [12.2. 环境管理](#122-环境管理)
    - [12.2.1. venv](#1221-venv)
    - [12.2.2. virtualenv](#1222-virtualenv)
    - [12.2.3. virtualenvwrapper](#1223-virtualenvwrapper)
- [python 解释器](#python-解释器)
  - [python Environment variables](#python-environment-variables)
  - [](#)

# 1. Python 的背景

**优点**:

1. 语法简单
   1. Python 不要求在每个语句的最后写分号, 当然写上也没错；
   2. 定义变量时不需要指明类型, 甚至可以给同一个变量赋值不同类型的数据。
   3. 这两点也是 PHP、JavaScript、MATLAB 等常见脚本语言都具备的特性。
2. Python 是开源的
3. Python 是免费的
4. Python 是高级语言
   1. Python 封装较深, 屏蔽了很多底层细节
   2. Python 会自动管理内存（需要时自动分配, 不需要时自动释放）
5. 解释型语言, 能跨平台
6. 是面向对象的编程语言
7. Python 功能强大（模块众多）
8. 可扩展性强
   1. 类库的底层代码不一定都是 Python, 还有很多 C/C++ 的身影。
   2. 当需要一段关键代码运行速度更快时, 就可以使用 C/C++ 语言实现, 然后在 Python 中调用它们
   3. 依靠其良好的扩展性弥补了运行效率慢的缺点

**缺点**:

1. 运行速度慢是解释型语言的通病
   1. Python 的运行速度几乎是最慢的, 不但远远慢于 C/C++, 还慢于 Java
2. 代码加密困难

当代应用场景速度不是第一要求

1. 多花钱就可以堆出高性能的硬件, 硬件性能的提升可以弥补软件性能的不足
2. 比如网站, 用户打开一个网页的大部分时间是在等待网络请求(500ms) , 而不是等待服务器执行网页程序(20ms)

## 1.1. 版本区别  2和3

1. Python 3.x print函数代替了print语句
   - Python2.x 中, 输出数据使用的是 Print 语句 `print "3,4"`
   - Python 3.x 中, print 语句没有了, 取而代之的是 print 函数 `print(3,4)`
2. Python 3.x 默认使用 UTF-8 编码
   - Python 2.x 默认采用的 ASCII 编码
   - Python 3.x 默认使用 UTF-8 编码
3. Python 3.x 除法运算
   - 使用运算符 `/` 除法运算, 结果也会是浮点数
   - 运算符 `//` 进行的除法运算叫做 floor 除法（向下取整）。
4. Python 3.x 八进制字面量表示
5. 异常抛出
   - 在 Python 2.x 版本中, 所有类型的对象都是直接被抛出的
   - 只有继承 BaseException 的对象才可以被抛出
6. Python 3.x 不等于运算符
   - 在 Python 3.x 中去掉了 <>, 只有 ！=

## 1.2. Python PEP文档

PEP（Python Enhancement Proposal）, 全称是 Python 改进方案  

有以下 3 个用途：

    通知：汇总 Python 核心开发者重要的信息, 并通过 Python 发布日程；
    标准化：提供代码风格、文档或者其他指导意见；
    设计：对提交的功能进行说明。

## 1.3. 底层语言

1. Python, 又称为 CPython。平时我们所讨论的 Python, 指的其实就是 CPython , 是用 C 语言编写实现的
2. 用 Java 语言实现的 Python 称为 JPython
3. 用 .net 实现的 Python 称为 IronPython
4. PyPy 中, Python 解释器本身是用 Python 编写的

# 2. Python的语法

1. 标识符的命名和C没有区别, 但是以下划线开头的标识符有特殊含义
   - 单下划线开头的标识符`_width`, 表示不能直接访问的类属性, 其无法通过 `from...import*`的方式导入
   - 双下划线开头的标识符`__add`, 表示类的私有成员
   - 双下划线作为开头和结尾的标识符`__init__`, 是专用标识符
2. python 的字符串可以用单引号`'ABC'`**或者**双引号表示`"ABC"`
3. `#` 号代表单行注释, `""" 注释 """` 三个双引号中间的内容代表多行注释 (其实是长字符串)
4. python代码第一行的声明 需要注明解释器脚本, 这样可以在命令行中直接调用脚本
   - windows 下 `#! python3`
   - OS X 下, `#! /usr/bin/env python3`
   - Linux 下, `#! /usr/bin/python`
5. python 使用缩进和冒号`:` 来分隔代码块
   - Python 中实现对代码的缩进, 可以使用空格或者 Tab 键实现。
   - 同一个级别代码块的**缩进量**必须一样, 否则解释器会报 SyntaxError 异常错误
   - 具体缩进量为多少, 并不做硬性规定
   - Tab 键, 通常情况下都是采用 4 个空格长度作为一个缩进量
   - （默认情况下, 一个 Tab 键就表示 4 个空格）
   - 缩进规则在大多数情况下有用,可以使用 `/` 来使得一条指令可以跨越到下一行,同时使下一行的缩进无效,增加代码可读性  

- 程序运行中使用<kbd>Ctrl</kbd>+<kbd>C</kbd> 可以立即终止程序

## 2.1. Python 书写规范 (PEP 8)

1. 每个 import 语句只导入一个模块, 尽量避免一个语句导入多个模块
2. 每行代码不需要分号, 也不要用分号将两条命令放在同一行
3. 每行不超过 80 个字符, 如果超过, 建议使用小括号将多行内容隐式的连接起来, 而不推荐使用反斜杠 `\` 进行连接

**必要的空行可以增加代码的可读**

1. 顶级定义（如函数或类的定义）之间空两行
2. 而方法定义之间空一行
3. 运算符两侧、函数参数之间以及逗号两侧, 都建议使用空格进行分隔

## 2.2. Python 保留字

Python 包含的保留字可以执行命令进行查看

```py
import keyword
keyword.kwlist
```

- 包保留字
  - import  : 导入
  - as      : 重命名
  - from    : 母包

- 类型保留字
  - True False              : bool类型的值  
  - None                    : NoneType的值

- 函数保留字
  - def  : 函数定义
  - lambda : lambda函数
  - yield : 定义生成器

- 运算保留字
  - and or not is           : 逻辑保留字
  - not 可以和一些保留字组合取反义

- pass                    : 空语句

- 控制保留字
  - for while elif if else  : 流程控制
  - break continue          : 循环控制

- assert                  : 断言
- in                      : 序列元素检查

- del  : 字典删除键值对
- class : 创建类

其他保留字:  
finally
global  
nonlocal
raise  
return  
with  

- 鲁棒性保留字
  - try  
  - except  

## 2.3. python 的类型提示

- python 本身的运行时不强制执行函数和变量类型注解
- 但是加入类型注解可以辅助 IDE 等第三方工具的错误检查

详情查看 typing 包 [相对链接](../pystl/devtools.md#typing)


- 把类型赋予一个别名, 相当于 typedef
- 可以用于简化复杂的类型签名
- 输入参数和返回值都可以进行注解

```py
# 将 Vector 代表一个浮点数的列表
Vector = list[float]

def greeting(name: str) -> str:
    return 'Hello ' + name
```

## 2.4. 作用域

同C语言 全局和局部的覆盖原则也相同  
函数中引用外部变量 使用`global` 关键字

```python
spam=1
def fun()
    global spam
    spam=2
```

作用域规则

1. 如果一个变量在**所有函数**之外, 它就是全局变量
2. 如果一个函数中
   1. 有针对变量的global语句, 则他是全局变量  
   2. 否则, 变量用于函数中的赋值语句, 它就是局部变量
   3. 但是若该变量没有用在赋值语句, 则仍然是全局变量
3. 在同一个函数中, 同一个变量要么总是全局变量, 要么总是局部变量, 不能改变

例如  

```python
def fun():
  eggs='123' #局部
def fun():
   print(eggs) #全局
def spam():
  print(eggs) #全局  
  eggs='spam local' #局部
  #在这里出错
```

# 3. 内置函数 Build-in Function

可以直接使用, 不需要导入某个模块, 解释器自带的函数叫做内置函数  

    在 Python 2.x 中, print 是一个关键字；到了 Python 3.x 中, print 变成了内置函数。

不要使用内置函数的名字作为标识符使用, 虽然这样做 Python 解释器不会报错, 但这会导致同名的内置函数被覆盖, 从而无法使用  

- type() : 输出变量或者表达式的类型
- id()  : 获取变量或者对象的内存地址
- help(): 输出用户定义参数对象的说明文档

执行一个字符串形式的 Python 代码:  

- exec(source, globals=None, locals=None, /) : 执行完不返回结果
- eval(source, globals=None, locals=None, /) : 执行完要返回结果

导入包的内置函数, 用于处理名称里带空格或者首字母是数字的模块

- `__import__()`

其他内置函数

```
abs()  delattr()  hash()    
all()      min()  setattr()
any()  dir()    next()  slicea()
ascii()  divmod()   object()  
      staticmethod()
     open()  
breakpoint()     sum()
  filter()  issubclass()  pow()  super()
    iter()  
callable()  format()  len()  property() 
frozenset()     
classmethod()  getattr()  
compile()    map()   
complex()  hasattr()  max()  round()
```

## 3.1. 类函数

这些内置函数实际上是内置类的定义函数, 具体用法查看下文的内置类  

- dict()
- list()
- tuple()
- set()
- 二进制序列
  - bytes()
  - bytearray()
  - memoryview()

## 3.2. 类型转换函数

虽然 Python 是弱类型编程语言, 在一些特定场景中, 仍然需要考虑到类型  

- 确认一个变量的类型
  - isinstance(object, classinfo)   如果 object 的类型属于 classinfo 中的一种, 返回 True

1. 因为python默认读入的内容都被识别为字符串, 因此需要类型转换
   - int(x,n)  将 x 转换成整数类型
   - float(x)  将 x 转换成浮点数类型
   - bool()

2. 使用print()输出时, 通过一行语句输出字符串和变量内容但不使用格式化字符串时, 需要转换
   - str(x)  将 x 转换为字符串
   - repr(x)  将 x 转换为表达式字符串

3. ASCII码函数
   - ord() : 字符转化成 Unicode 码
   - chr(integer ) : 数字的 Unicode 码 转化成字符

4. 进制转换
   - 除了转换成10进制的数, 其他的进制转换结果都带有前缀, 可以使用切片去除前缀
   - hex(x,n)  : 将一个整数 x 转换为一个十六进制字符串, n代表原本的进制
   - oct(x,n)  : 将一个整数 x 转换为一个八进制的字符串, n代表原本的进制
   - int(x,n)  : 将一个整数或 (字符串,bytes,bytearray) x 转换为整数, n代表原本的进制
   - bin(x,n)  : 将一个整数 x 转换为一个二进制的字符串, n代表原本的进制

## 3.3. 作用域变量获取函数

- vars(object) : 返回object 的 `__dict__` 属性, 如果object没有该属性则报异常
- locals()    : 将局部空间的所有变量以字典形式返回, 等同于 `vars(空)`
- globals()   : 将全局空间的所有变量以字典形式返回, 注意以本 module 为基准, 不包括调用的module

## 3.4. print()

`print (value,...,sep=' ',end='\n',file=sys.stdout,flush=False)`  

- value 参数可以接受任意多个变量或值, 因此 print() 函数可以输出多个值
- `end` 参数的默认值是 `\n`
- `sep` 参数的默认值是 `` 空格
- `file` 参数的默认值为 sys.stdout, 代表标准输出, 即屏幕输出  
- `flush` 参数用于控制输出缓存, 该参数一般保持为 `False` 即可, 这样可以获得较好的性能

```py
print("读者名：",user_name,"年龄：",user_age)

#打开文件
f = open("demo.txt","w")
print('123456',file=f)
```

### 3.4.1. 格式化输出

类似于C语言的格式化输出, print() 函数提供了类似的功能

- 在 print() 函数中, 由引号包围的是格式化字符串, 它相当于一个字符串模板
- 格式化字符串中的`%` 是转换说明符（Conversion Specifier）只是一个占位符
- 转换说明符会被后面表达式替代
- `%` 也被用作分隔符, 格式化字符串后接一个`%` 再接输出的表达式
- 格式化字符串中可以包含多个转换说明符, 这个时候也得提供多个表达式, 此时多个表达式必须使用小括号( )包围起来

```py
name = "C语言中文网"
age = 8
url = "http://c.biancheng.net/"
print("%s已经%d岁了, 它的网址是%s。" % (name, age, url))
```

转换说明符的各种格式和C语言基本一致
| 转换符 | 解释方法                               |
| ------ | -------------------------------------- |
| %d、%i | 转换为带符号的十进制整数               |
| %o     | 转换为带符号的八进制整数               |
| %x、%X | 转换为带符号的十六进制整数             |
| %e     | 转化为科学计数法表示的浮点数（e 小写） |
| %E     | 转化为科学计数法表示的浮点数（E 大写） |
| %f、%F | 转化为十进制浮点数                     |
| %g     | 智能选择使用 %f 或 %e 格式             |
| %G     | 智能选择使用 %F 或 %E 格式             |
| %c     | 格式化字符及其 ASCII 码                |
| %r     | 使用 repr() 函数将表达式转换为字符串   |
| %s     | 使用 str() 函数将表达式转换为字符串    |

### 3.4.2. 控制输出方式

转换说明符的`%`和类型字符中间可以加入控制内容

- 数字指定最小输出宽度  `%10d`
  - 当数据的实际宽度小于指定宽度时, 会在左侧以空格补齐
  - 当数据的实际宽度大于指定宽度时, 会按照数据的实际宽度输出, 不会裁剪
- 对齐标志指定对齐方法, 加在`数字之前`, 默认是左侧补空格, 数据靠右边
  - `-` 指定左对齐,
  - `+` 指定数字数据带正负号 `+-`, 只对数字数据生效
  - `0` 指定数据用 0 代替空格补足, 只对数字数据生效
    - 左对齐时因为右侧补零会改变整数的数值, 因此左对齐时不对整数生效
- 对小数指定精度, 放在`最小宽度数据之后, 用点号隔开`  
  - `%m.nf` `m` 表示最小宽度, `n` 表示输出精度, `.`是必须存在的, f表示小数输出

```py
n = 1234567
print("%10d." % n)
print("%5d." % n)

f = 3.141592653
# 最小宽度为8, 小数点后保留3位
print("%8.3f" % f)
# 最小宽度为8, 小数点后保留3位, 左边补0
print("%08.3f" % f)
# 最小宽度为8, 小数点后保留3位, 左边补0, 带符号
print("%+08.3f" % f)


```

## 3.5. input()

字符串形式接受用户输入  

`str = input(tipmsg)`  

- str   : 表示输入存入的变量
- tipmsg: 表示在控制台中输出的提示信息, 提示输入什么内容

## 3.6. open() 基础文件操作

open() 函数用于创建或打开指定文件, 返回一个 `file object`  
`open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None)`  

- file    : 一个路径, 最好是 path-like oject
- mode    : 文件模式
- buffering: 设置 buffering policy, 一个整数.
- encoding: 只能在文本模式下使用, 用字符串代表编码格式
  
文件模式

- r读取 w写入 +读写 a谁家
- b二进制, t 文本
- x 如果文件已存在则失败

file object 是python的一个内置类, 包含了一些基础函数用于读写文件
### 3.6.1. read

* f.read(size)
* f.readline()


如果是按行来读取文件, 可以使用 for 循环
This is memory efficient, fast, and leads to simple code:
```py
for line in f:
  print(line, end='')
```
### 3.6.2. write

* f.write(str)

### 3.6.3. position

* f.tell()
* f.seek(offset,whence)

## 3.7. range()

- range() 函数能够轻松地生成一系列的数字  
- 函数的返回值并不直接是列表类型 list  而是 range
- 如果想要得到 range() 函数创建的数字列表, 还需要借助 list() 函数

```py
# 参数分别是 
# 初始值
# 结束值 , 数字列表中不会包含该值, 是开区间
# 步长
even_numbers = list(range(2,11,2))

```

## 3.8. zip()

- 可以将多个序列（列表、元组、字典、集合、字符串以及 range() 区间构成的列表）“压缩”成一个 zip 对象
- 所谓“压缩”, 其实就是将这些序列中对应位置的元素重新组合, 生成一个个新的元组
- 函数“压缩”多个序列时, 它会分别取各序列中第 1 个元素、第 2 个元素、... 第 n 个元素, 各自组成新的元组
- 需要注意的是, 当多个序列中元素个数不一致时, 会以最短的序列为准进行压缩

```py
# iterable,... 表示多个列表、元组、字典、集合、字符串, 甚至还可以为 range() 区间
zip(iterable, ...)


my_list = [11,12,13]
my_tuple = (21,22,23)
print([x for x in zip(my_list,my_tuple)])
# [(11, 21), (12, 22), (13, 23)]

my_pychar = "python"
my_shechar = "shell"
print([x for x in zip(my_pychar,my_shechar)])
# [('p', 's'), ('y', 'h'), ('t', 'e'), ('h', 'l'), ('o', 'l')]

```

## 3.9. reserved()

- `reversed(seq)` 并不会修改原来序列中元素的顺序
- 对于给定的序列（包括列表、元组、字符串以及 range(n) 区间）, 该函数可以返回一个逆序列表的`迭代器`

```py
print([x for x in reversed([1,2,3,4,5])])
# [5, 4, 3, 2, 1]


print([x for x in reversed((1,2,3,4,5))])
# [5, 4, 3, 2, 1]
# 逆序元组, 返回的还是一个列表
```

## 3.10. sorted()

- `list = sorted(iterable, key=None, reverse=False)`
- iterable 表示指定的序列
- key 参数可以自定义排序规则
- reverse 参数指定以升序（False, 默认）还是降序（True）进行排序
- sorted() 函数会返回一个排好序的**列表**
- 字典默认按照key进行排序

```py
a = "51423"
print(sorted(a))
# ['1', '2', '3', '4', '5']


chars=['http://c.biancheng.net',\
       'http://c.biancheng.net/python/',\
       'http://c.biancheng.net/shell/',\
       'http://c.biancheng.net/java/',\
       'http://c.biancheng.net/golang/']
#自定义按照字符串长度排序
# 传入 key 等于一个 lambda 函数
print(sorted(chars,key=lambda x:len(x)))
```

## 3.11. enumerate() 遍历对象函数

- 一般用在 for 循环中, 将一个可遍历的数据对象组合成一个索引序列  
- 可以同时列出数据和数据下标

```py
# start 代表起始位置
enumerate(sequence, [start=0])

# 返回枚举对象
seq = ['one', 'two', 'three']
for i, element in enumerate(seq):
   print i, element
```

# 4. Python的操作符与变量

Python是弱类型语言

- 变量无须声明就可以直接赋值
- 变量的数据类型可以随时改变
- 使用 type() 内置函数类检测某个变量或者表达式的类型

## 4.1. 基础类型

整数, 浮点数, 复数, 字符串, bytes, 布尔

- 整数不分类型, Python 整数的取值范围是无限的
  - 当所用数值超过计算机自身的计算能力时, Python 会自动转用高精度计算（大数计算）
  - 为了提高数字的可读性, 可以用下划线来分割数字而不影响本身的值 `1_000`
  - 可以使用多种进制表示数字
    - 十进制形式的整数不能以 0 作为开头, 除非这个数值本身就是 0
    - 二进制形式书写时以`0b`或`0B`开头
    - 八进制形式以`0o`或`0O`开头
    - 十六进制形式书写时以`0x`或`0X`开头
- 浮点数也只有一种类型
  - 书写小数时必须包含一个小数点, 否则会被 Python 当作整数处理`34.6`
  - 用指数形式生成的值也是小数 `2e2`= 200
- Python 语言本身就支持复数
  - 复数的虚部以j或者J作为后缀 `a + bj`
  - 本身支持简单的复数运算
- 字符串必须由双引号`" "`或者单引号`' '`包围
  - 字符串中的引号可以在引号前面添加反斜杠`\`就可以对引号进行转义
  - 普通字符串或者长字符串的开头加上`r`前缀, 就变成了原始字符串
  - 如果字符串内容中出现了单引号, 那么我们可以使用双引号包围字符串, 反之亦然
  - 使用三个单引号或者双引号其实是 Python 长字符串的写法
    - 所谓长字符串, 就是可以直接换行, 可以在字符串中放置任何内容, 包括单引号和双引号
    - 长字符串中的换行、空格、缩进等空白符都会原样输出
- bytes 类型用来表示一个二进制序列, 类似于无类型的指针
  - 只负责以字节序列的形式（二进制形式）来存储数据, 至于这些数据到底表示什么内容, 完全由程序的解析方式决定
  - bytes 只是简单地记录内存中的原始数据, 至于如何使用这些数据, bytes 并不在意
  - bytes 是一个类, 调用它的构造方法, 也就是 bytes(), 可以将指定内容转化成字节串
  - 如果字符串的内容都是 ASCII 字符, 那么直接在字符串前面添加b前缀就可以将字符串转换成 bytes
- 字符串有一个 encode()  将字符串转换成bytes()
- bytes 有一个 decode() 将字节串转换成字符串
- bool 类型用来表示真或者假, 真假都是python的保留字
  - True
  - False
  - 对于整型的`0`,浮点的`0.0`,字符串的 `''` 都被认为是`False` , 其余为 `True`
  - `[ ]`  #空列表
  - `( )`  #空元组
  - `{ }`  #空字典
  - `None`  #空值
  - 以上都被认为是 False
- `None` 是NoneType类型的唯一值,类似于C语言的 `null`

## 4.2. 转义字符

在 Python 中, 一个 ASCII 字符除了可以用它的实体（也就是真正的字符）表示, 还可以用它的编码值表示  

- Python 转义字符只对 ASCII 编码（128 个字符）有效
- 转义字符以`\0`或者`\x`开头
- `\0`开头表示后跟八进制形式的编码值
- `\x`开头表示后跟十六进制形式的编码值
- 转义字符只能使用八进制或者十六进制
- ASCII 编码共收录了 128 个字符, `\0`和`\x`后面最多只能跟两位数字, 所以八进制形式\0并不能表示所有的 ASCII 字符, 只有十六进制形式\x才能表示所有 ASCII 字符

针对常用的控制字符, Python使用了C语言定义的简写方式

- \n  换行符, 将光标位置移到下一行开头。
- \r  回车符, 将光标位置移到本行开头。
- \t  水平制表符, 也即 Tab 键, 一般相当于四个空格。
- \a  蜂鸣器响铃。注意不是喇叭发声, 现在的计算机很多都不带蜂鸣器了, 所以响铃不一定有效。
- \b  退格（Backspace）, 将光标位置移到前一列。
- `\\`  反斜线
- \'  单引号
- \"  双引号
- \  在字符串行尾的续行符, 即一行未完, 转到下一行继续写。

## 4.3. 运算符

1. 算术运算符
   - `+ - * %`与C语言相同
   - `/` 除法返回小数
   - `//` 除法取商 返回整数
   - `**` 代表求指数 2**8=256

2. 位运算符  位运算符只能用来操作整数类型
   - `& | ~`  与或非运算符
   - `^` 按位异或运算
   - `<< >>` 左移 右移

3. 赋值运算符 `=`
   - 赋值表达式也是有值的, 它的值就是被赋的那个值, 可以使用 `a = b = c = 100` 同时赋值三个变量
   - 和以上所有运算符结合, 效果和C语言一样 `+=  -=  |=  &=`

4. 比较运算符
   - `== != > < >= <=` 都与C语言相同
   - `==`和`!=`可以用于所有类型,若两边类型不同则永远不会相等
   - `is`  判断两个变量所引用的对象是否相同, 如果相同则返回 True, 否则返回 False。
   - `is not`  判断两个变量所引用的对象是否不相同
   - is 运算不单单是值相同, 还可以理解为地址相同, 同一个对象

5. 逻辑运算符
   - **Boolean**型的值只能为 `True` 和 `False` 都是保留字没有单引号,首字母必须大写
   - 逻辑运算与C语言的操作符不同
   - && -> `and` , || -> `or` , ! -> `not`     顺序  not>and>or
   - Python 逻辑运算符可以用来操作任何类型的表达式, 不管表达式是不是 bool 类型
   - 同时, 逻辑运算的结果也不一定是 bool 类型, 它也可以是任意类型
   - and 和 or 运算符会将其中一个表达式的值作为最终结果, 而不是将 True 或者 False 作为最终结果

6. 三目运算符
   - Python 是一种极简主义的编程语言, 它没有引入? :这个新的运算符
   - 使用已有的 if else 关键字来实现相同的功能
   - `exp1 if contion else exp2`
   - `max = a if a>b else b`

# 5. python Built-in Types

Built-in Types 代表那些包括在了 python 解释器里的变量, 不需要任何包(包括STL), 即可正确处理

主要的 built-in type 可以按照以下分类:
* numerics
* sequences
* mappings
* classes
* instances
* exceptions

这些数据类型包括集合类型, 也有成员可变集合类型

所有的内置成员变量都实现了以下功能
* 可以互相之间比较 equality
* Truth Value Testing
* 转换成 string 类型, 包括
  * str()  , 作为 print() 的参数被输出时会隐式调用
  * repr()

## 5.1. 序列

序列是Python所有核心序列的统称, 包括字符串、列表、元组、集合和字典  
全部支持: 存在检查, 遍历  

- 标准序列: 字符串 str 、列表 list 、元组 tuple , 支持索引、切片、相加和相乘
- 二进制序列: bytes
- 广义序列: 集合和字典,        不支持索引、切片、相加和相乘

### 5.1.1. 序列通用
#### 5.1.1.1. 序列内置函数

因为是序列的内置函数, 因此字符串列表元组字典集合都有这些函数  

| 函数        | 功能                                          |
| ----------- | --------------------------------------------- |
| len()       | 计算序列的长度, 即返回序列中包含多少个元素。  |
| max()       | 找出序列中的最大元素。                        |
| min()       | 找出序列中的最小元素。                        |
| list()      | 将序列转换为列表。                            |
| str()       | 将序列转换为字符串。                          |
| sum()       | 计算元素和。                                  |
| sorted()    | 对元素进行排序。                              |
| reversed()  | 反向序列中的元素。                            |
| enumerate() | 将序列组合为一个索引序列, 多用在 for 循环中。 |

- 注意, 对序列使用 sum() 函数时, 做加和操作的必须都是数字, 不能是字符或字符串, 否则该函数将抛出异常
- 因为解释器无法判定是要做连接操作（+ 运算符可以连接两个序列）, 还是做加和操作。

#### 5.1.1.2. 序列的引用

对于变量保存字符串和整数值, 变量保存的是值本身  
对于列表等序列对象, 变量保存的是列表引用, 类似于指针  

```py
spam=[1,2,3]
cheese=spam
```

1. 在函数调用时, 对于列表参数是传递引用, 因此需要注意可能的缺陷
2. copy模块,有时候为了不影响原来的列表或字典,使用copy模块,确保复制了列表
   - `spam=['A','B','C']`
   - `cheese=copy.copy(spam)`

#### 5.1.1.3. 存在检查 in

`value in sequence`  
可以和 not 结合 `not in`

```py
str="c.biancheng.net"
print('c'in str) # True
```

#### 5.1.1.4. 索引

- 最基础的操作, 方括号`[]`配合数字访问元素  
- python是 `0索引`
- python支持负数索引,  `[-n]` 等同于 `[len()-n]`

#### 5.1.1.5. 切片

访问一定范围内的元素, 通过切片操作生成一个新的对应序列  

`sname[start : end : step]`  

- sname : 序列
- start : 开始索引位置, 包括该位置 ,默认0
- end   : 结束索引位置, **不包括该位置**, 默认为序列的长度
- step  : 切片间隔, 默认为1, 默认值的时候可以省略第二个冒号`:`

- 切片下标的省略`[:2] [1:] [:}` 分别代表从头开始,到末尾,全部

#### 5.1.1.6. 序列相加

两种**类型相同**的序列使用 `+` 运算符做相加操作  
它会将两个序列进行**连接**, 但不会去除重复的元素  

#### 5.1.1.7. 序列相乘

使用数字 `n` 乘以一个序列会生成新的序列, 其内容为原来序列被重复 `n` 次的结果

### 5.1.2. 列表类型 list

- 列表类型是最基础的序列类型, 其他的序列类可以用 `.list()` 转化为 list  
- 所有序列都有 `.list()` 成员就可想而知列表的基础性
- Python的列表取代了其他语言的数组  

- 列表值: 指的是列表本身,可以作为值保存在变量中,或者传递给函数
- 表项:   指的是列表内的值,用逗号分隔
- 使用下标访问列表中的每个值 `spam[0]` 若超出列表中值的个数,将返回 `IndexError`
- 列表中包含其他的列表值,构成多重列表,使用多重下标访问 `spam[0][1]`
- 负数下标: 使用负数来代表倒数的索引, `[-1]` 代表列表中倒数第一下标

#### 5.1.2.1. 列表的创建使用

python的列表总是动态的

```python
# 使用 [ ] 直接创建列表
catName=[]
while True:
    catName=catName+[name]
for name in catName:
    print(name)


# 使用内置函数 list() 函数创建列表
list1 = list("hello")
print(list1)

#将字典转换成列表会丢失vlaue
dict1 = {'a':100, 'b':42, 'c':9}
list3 = list(dict1)
# ['a', 'b', 'c']
print(list3)
```

#### 5.1.2.2. 元素添加

1. 序列中提到的, `+` 可以连接多个列表, 运算返回新列表不会改变原有列表, 但是效率低
2. .append(obj) 成员函数在末尾添加元素, obj 可以是元组或列表但都会被当做一整个元素
3. .extend(obj) 成员函数在末尾添加元素, 会读取 obj 内部的单个元素一一添加
4. .insert(index,obj) 在指定位置插入元素, obj 是一整个元素

```py
l = ['Python', 'C++', 'Java']
t = ('JavaScript', 'C#', 'Go')

l.append(t)
# ['Python', 'C++', 'Java', 'PHP', ('JavaScript', 'C#', 'Go')]

l.extend(t)
# ['Python', 'C++', 'Java', 'C', 'JavaScript', 'C#', 'Go']

l.insert(1, 'C')
# ['Python', 'C', 'C++', 'Java']
```

#### 5.1.2.3. 元素删除

1. del 关键字, 专门执行删除操作
   - 可以删除整个列表, 但是python有垃圾自动整理, 不常用
   - 可以删除列表单个元素 `del listname[index]`
   - 可以删除切片 `del listname[start : end]`
2. .pop(index) 成员函数, 删除指定索引处的元素并返回
   - 注: python 没有提供 push 函数, pop 也属于广义上的 pop
   - 不提供参数的时候, 和普通的 pop() 一样返回的是末尾的函数
3. .remove(value) 成员函数 , 按照值来删除元素
   - 只会删除第一个和指定值相同的元素
   - 必须保证元素是存在的, 否则会引发报错 `ValueError`
4. .clear() 清空列表

总结:

- 若知道删除的下标,就用 `del spam[2]`
- 在删除的同时返回删除的值,就用`a=spam.pop(2)`
- 若知道要删除的值,就用`spam.remove('a')`

#### 5.1.2.4. 元素修改

1. 用索引直接修改
2. 用切片直接修改多个元素, 但是必须保证输入的元素个数和切片的元素个数相同

```py
nums = [40, 36, 89, 2, 36, 100, 7]
nums[2] = -26  #使用正数索引
nums[-3] = -66.2  #使用负数索引

nums[1: 4] = [45.25, -77, -52.5]

# 带步长的切片也可以使用, 确保个数相同比较麻烦
nums[1: 6: 2] = [0.025, -99, 20.5]
```

#### 5.1.2.5. 列表的查找

1. 除了序列提供的 in 关键字操作, 列表查找元素还有其他方法
2. .count(obj) 统计元素obj在列表中的出现个数
3. .index(obj, start,end) 在范围[start,end) 中查找元素出现的位置
   - 该方法非常脆弱 , 元素不存在的话会报错 `ValueError`
   - 使用前最好用 count() 统计一下

#### 5.1.2.6. 列表的赋值

使用列表可以同时为多个变量赋值  
`size,color,dispo=cat`  
变量的数目和列表的长度必须相等,否则报错  
`cat=['fat','black','loud']`

#### 5.1.2.7. 增强的赋值操作

针对`+、-、*、/、%`操作符  
有`+=、-=、*=、/=、%=`意义同C语言

```python
spam +=1 
spam=spam+1
```

`+=`可以完成字符串或列表的连接  

```python
spam='hello'  
spam+='world'
```

`*=`可以完成字符串或列表的复制  

```python
bacon=['zon']
bacon*=3
```

#### 5.1.2.8. 列表类型的方法

- `remove()`: `spam.remove('bat')` 传入一个值, 将该值从列表中删除, 若该值不在列表中, 返回`ValueError`错误,只会删除第一次出现的位置
  *
- `sort()`:
  - `spam.sort()`数值的列表或者字符串的列表可以使用该方法
  - `spam.sort(reverse=True)` 指定反向排序
  - `sort()` 是当场排序,不是返回值,不能对混合值列表进行排序
  - `sort()` 是按照ASCII表排序,因此在大写是排在小写字母之前
    - 可以使用`spam.sort(key=str.lower)`使方法把所有的表项当作小写
- 函数`sorted()`
  - 可以返回一个排序的值,但是不会影响原本的值 `ab=sorted(spam)`

### 5.1.3. 元组

元组(tuple)是另一个重要的序列, 和列表非常类似

- 列表的元素是可以更改的
- 元组一旦创建就不可更改, 不可变序列

- 转换成元组的函数是 `tuple(data)`  
- data 表示可以转化为元组的数据, 包括字符串、元组、range 对象等。
- data 为字典时转化后会丢失 value

#### 5.1.3.1. 元组创建

元组与列表几乎一样,除了两点  

1. 元组输入时使用`()`而不是`[]` 例:`eggs=('hello',42,0.5)`
2. 元组属于不可变类型, 同字符串一致,即值不能增改,字符串属于元组的一种
3. 为了区分只有一个值的元组与与普通的值,在值的后面添加一个 `,`
4. 元组通常都是使用一对小括号将所有元素包围起来的, 但小括号不是必须的, 只要将各元素用逗号隔开, Python 就会将其视为元组

```py
type ( ('hello',) )
# <class 'tuple'>

type( ('hello') )
# <class 'str'>

course = "Python教程", "http://c.biancheng.net/python/"
# ('Python教程', 'http://c.biancheng.net/python/')
print(course)
```

#### 5.1.3.2. 元组的访问和修改

1. 作为标准序列之一, 支持索引和切片
2. 元组作为不可变序列, 修改时只能用新的元组代替旧的元组
3. 删除时也只能用 `del` 直接删除整个元组

### 5.1.4. 二进制序列 bytes

专门用于处理二进制数据的类型  
包括 bytes, bytearray, memoryview 三种细微不同的子类
* bytes       : 二进制对象的存储类, 具有不可改变性, 类似于元组
* bytearray   : 相比于 bytes, 赋予了可更改的特性
* momoryview  : 用于直接映射到 内存中的二进制对象, 从而不需要拷贝就可以访问

#### 5.1.4.1. bytes


- 非可变类型, 存储单一 byte 的序列
- 拥有语法定义, 可以通过类似于字符串的方式被创建, 但是引号前要加一个字母 b `b"abc"`, 对三种字符串语法都生效
- 只有 ASCII 的字符可以直接转换成 bytes

`class bytes([source[, encoding[, errors]]])`

* 因为一个 byte 可以被表示成 2个16进制数, 所以有很多情况下 二进制数据被16进制数来保存或存储
* 类方法 `bytes.fromhex(str)`
  * 从一个被字符串化的 16进制序列还原出一个 bytes 对象
* 对象方法 `hex([sep[, bytes_per_sep]])`
  * 把一个 bytes 对象转换成 16进制数表示的字符串
  * sep 是每个 byte 之间的间隔符, 一般是空格类
  * bytes_per_sep 表示间隔符的出现频次, 默认是1, 正负号代表从右左开始统计

#### 5.1.4.2. bytearray

- 可变类型
- 无语法定义, 必须用构造函数创建

`class bytearray([source[, encoding[, errors]]])` 
- `bytearray()` 创建空 bytearray
- `bytearray(10)` 创建 bytearray 并给每个 byte 填入 0
- `bytearray([1,255])` 创建对应内容的 bytearray 需要给每个字节填入数据 1~255, `bytearray(b'\x01\xff')`

<!-- ### bytes类 对象方法 -->



## 5.2. 字典 dict

- 类似于 C++ 的 map  但是字典是无序的, 没有**第一个**的概念
- 字典的键可以使用许多不同的数据类型
  - 不只是整数, 可以是字符串或者元组, 但是不能是列表
  - 键定义后即不能改变
- 可以任意深度的嵌套

- dict 的 in 和 not in 运算都是基于  key 来判断的
  
### 5.2.1. 字典创建删除

基本定义方法:

- 所有元素放在大括号`{}` 中
- 键和值之间使用冒号 `:` 分隔
- 相邻元素之间使用逗号 `,` 分隔

函数定义方法:

- dict.fromkeys(list,value=None) 带有默认值的创建
- dict() 映射函数, 默认创建空字典
  - a = dict(str1=value1, str2=value2) str 表示字符串的键, 必须是变量输入, 不能是引号输入
  - `demo = [('two',2), ['one',1]] a = dict(demo)` 传入一系列二元列表或元组
  - keys = 序列 values =序列  `a = dict( zip(keys, values) )`
  - dict() 和 zip(组合构成字典)

```py
# 标准创建
dictname = {'key':'value1', 'key2':'value2', ..., 'keyn':'valuen'}

#创建空字典
dict2 = {}

# 使用元组和数字作为key 键可以是不同类型
# 但是列表不能作为key, 因为列表天生是可变的
dict1 = {(20, 30): 'great', 30: [1,2,3]}

# 加入一个新的键值对:
ice_cream_1['flavor'] = 'vanilla'
# 删除一个键值对
del ice_cream_1['flavor'] 

# 带默认值的字典
knowledge = ['语文', '数学', '英语']
scores = dict.fromkeys(knowledge, 60)
# {'语文': 60, '英语': 60, '数学': 60}
print(scores)

```

### 5.2.2. 访问使用

字典有三个打印方法

- 这些方法返回的不是真正的列表(是 view) 不能被修改, 不能使用 append() 方法
- `keys()`  :返回键值
- `values()`:具体值
- `items()` :键-值对

- 但可以用于 `for` 循环  
- 可以使用 `list(spam.keys())` 将返回的值转换为列表  
- 类型分别为 dict_keys、dict_values 和 dict_items

安全访问方法: `get(key[, default])`

- 可以安全的通过索引访问一个值,当这个索引在字典中不存在时返回备用值  
- 若不使用`get`方法直接用索引会报错
- `print( a.get('one') )`

### 5.2.3. 成员函数操作

1. `.copy()`
   - 返回一个表层深拷贝的 dict
   - 即如果 value 是一个列表, 那么根据列表的引用原则, 列表的元素是新旧dict 共享内存的
2. `setdefault(key,defaultvalue)`
   - 可以安全的初始化一个键
   - 当该键存在则返回键值
   - 键不存在则设定为第二个参数, 并返回第二个参数
3. `.update(dict)`
   - 传入参数是一个 dict
   - 已有的key会被更新value
   - 不存在的key会被创建并赋值
4. `.pop(key)`
   - 弹出对应的键值对
5. `.popitem()`
   - 弹出dict内部的最后一个键值对
   - 表面上看是无序的

## 5.3. 集合 set

1. 集合（set）是一个无序的**不重复元素**序列, 相当于没有value的字典
2. 无法存储列表、字典、集合这些可变的数据类型

### 5.3.1. 创建

1. 通过花括号直接创建
2. 注意: 创建一个空集合必须用`set()` 而不是 `{ }`, 因为` { } `是用来创建一个空字典  
3. `set()` 函数实现
   - 将字符串、列表、元组、range 对象等可迭代对象转换成集合

```py
setname = {element1,element2,...,elementn}

a = {1,'c',1,(1,2,3),'c'}
```

### 5.3.2. 访问

- set 也是无序的, 因此不能使用下标

```py
a = {1,'c',1,(1,2,3),'c'}
for ele in a:
    print(ele,end=' ')
# 1 c (1, 2, 3)
```

### 5.3.3. 元素操作

1. 添加元素 `s.add( x )`
   - 如果 x 已经存在于集合, 则无改变,
   - `thisset.add("Facebook")`
2. 更高级的添加元素 `s.update( x )`
   - 这里 x 可以可以是列表, 元组, 字典等, x 还可以有多个用逗号分开
   - `thisset.update([1,4],[5,6])`
3. 移除元素 `s.remove( x )` 如果元素不存在就会错误`KeyError`
   - 另一个不会发生错误的移除元素方法 `s.discard( x )`
4. 清空集合 `s.clear()`
5. `pop()` 随机移除元素 并返回值

### 5.3.4. 集合运算

python的set作为一个集合, 可以运行数学上的集合运算  

- `&`  交集, 取公共元素
- `|`  并集, 取全部元素
- `^`  差集, A|B 取 A-A&B
- `-`  对称差集, 取 A和B 中不属于 A&B 的元素 A&B-A|B

## 5.4. 字符串 string

最基本的将object转为string的函数：

- str() 保留了字符串最原始的样子
- repr() 用引号将字符串包围起来, Python 字符串的表达式形式

字符串是一个 python 内建类:
* python的字符串使用 `''` 单引号输入, 也可以使用 `""`双引号输入, 区别在于使用双引号时字符串中可以包括单引号
* 所有的对应语句都会直接转换成字符串类, 并且可以使用对应的类方法


### 5.4.1. 字符串的输入

- 转义字符,使用`\` 反斜杠来输入特殊字符,  `\t` 制表位, `\n` 换行符, `\\` 反斜杠
- 原始字符串,在字符串前面加一个`r`,例如 `r'abc'` ,可以完全忽略字符串中的所有反斜杠
- 多行字符串,对于有换行的字符串可以更为方便的输入
- raw字符串,在字符串前加一个 'r' 代表rawString
   1. 取消对反斜杠\ 的转义
   2. 如果一个字符串包含很多需要转义的字符, 对每一个字符都进行转义会很麻烦。为了避免这种情况,我们可以在字符串前面加个前缀r, 表示这是一个 raw 字符串, 里面的字符就不需要转义了。
- 多行字符串`'''abc'''`,字符串可以跨过多行
   1. 三引号还会对其内的单双引号起作用
   2. 可以将raw字符串和多行字符串结合起来使用
- 字符串的编码类型
   1. python中字符串默认采用的ASCII编码,
   2. 如果要显示声明为unicode类型的话, 需要在字符串前面加上'u'或者'U' `print(u'字符串')`
   3. 如果中文字符串在Python环境下遇到 UnicodeDecodeError,在源码第一行添加注释`# -*- coding: utf-8 -*-`

```
>>> print(r'\n')
\n
```

```python
abc='''this is a long
string with multiline.
'''
```

字符串可以使用切片以及 `in`  `not in` 操作符,用来比较前一个字符串是否在后一个字符串中间
  
### 5.4.2. 字符串编码

字符串有一个方法 `.encode()` 可以将字符串使用对应格式进行编码, 返回一个对应的 bytes 类型

#### 5.4.2.1. 大小写及内容检测方法

- `upper()` 和 `lower()` 返回一个**新字符串**,将原本字符串中所有字母转变为大写/小写
- `name.title()`  标题方法, 将字符串的单词第一个字母大写  
- `isupper()` 和 `islower()` 返回布尔值,如果这个字符串不为空且全部字符为大写/小写则返回`True`
- 其他的 isX 方法,返回True的情况
  - isalpha()  非空且只包含字母
  - isalnum()  非空且只包含字母和数字
  - isdecimal() 非空且只包含数字字符
  - isspace()  非空且只包含空格,制表符,换行
  - istitle()  非空且只包含以大写字母开头,后面都是小写字母的 `单词` 及可以包含空格及数字

#### 5.4.2.2. 开头结尾检测方法  

   startswith() 和 endswith()
   以传入的字符串为开始/结尾,则返回True

#### 5.4.2.3. 组合与切割方法  

   join() 和 split()
    join()是将一个字符串列表连接起来的方法,并在字符串中间插入调用`join`的字符串  
    `','.join(['a','b','c'])`   返回 ` 'a,b,c' `  
    split()则相反,在被拆分的字符串上调用,默认以各种空白字符分隔  
    `'My name is'.split()`   返回 `['My','name','is']`  
    常被用来分割多行字符串  `spam.split('\n')`

#### 5.4.2.4. 对齐方法  

   rjust()  ljust()  center()
   在一个字符串上调用,传入希望得到的字符串的长度,将返回一个以空格填充的字符串  
   分别代表左对齐,右对齐  
   `'a'.rjust(5)`  返回  `'    a'`
   `'a'.ljust(5)`  返回  `'a    '`
   可以输入第二个参数改变填充字符  
   `'a'.ljust(5,'*')`  返回  `'a****'`

#### 5.4.2.5. 清除对齐方法  

   `strip()  rstrip() lstrip()`  
   在左右删除空白字符  
   传入参数指定需要删除的字符  注:这里第二个参数无视字符出现顺序  
   `'abcccba'.strip('ab')` 与
   `'abcccba'.strip('ba')` 作用相同

#### 5.4.2.6. 识别转换方法 **str.extract()**

和pandas组合使用,拆解字符串

```py

# Extracting ID into ['faculty','major','num']
id_ext = df['id'].str.extract('(\w{3})-(\w{2})(\d{2})',expand=True)
id_ext.columns = ['faculty','major','num']


# Extracting 'stats' into ['sex', 'age']
stats_ext = df['stats'].str.extract('(\w{1})_(\d{2})years',expand=True)
stats_ext.columns = ['sex', 'age']
```

#### 5.4.2.7. str.get_dummies()

为了对数据进行统计,例如多个学生选择了多种科目,科目可能会重复,这种统计  

```py
'''
name: course
Ali: Maths, English, Algebra, Geometry
Bea: Science, Biology, Physics, Chemistry, Maths, Music

'''
stud_course = pd.read_csv('student_course.csv', sep=":", skipinitialspace=True)


# Converting course_name into dummy variables
course_dummy_var = stud_course['course'].str.get_dummies(sep=', ')
print(course_dummy_var)

'''
  Algebra  Art  Biology   ...     Phys. Ed.  Physics  Science
0         1    0        0   ...             0        0        0
1         0    0        1   ...             0        1        1
'''

```

## 5.5. Iterator 

python 的迭代器概念: 
* 用户可以自己创建能够被迭代的自定义类
* 如下两个内部函数构成了 `iterator protocol`

`iterator.__iter__()`
* 返回该迭代器对象本身, 可以说用于证明该对象是迭代器, 可以用于初始化迭代的 count
* This method corresponds to the `tp_iter` slot of the type structure for Python objects in the Python/C API.

`iterator.__next__()`
* 返回下一个元素, 如果已经遍历完成, 唤起 `StopIteration` exception
  * 如果再次被调用 `__next__`, 必须持续的唤醒该 exception
* This method corresponds to the `tp_iternext` slot of the type structure for Python objects in the Python/C API.

`container.__iter__()`
* Return an iterator object.
* 每太看懂, 可能是所谓的 container 类的内部实现是通过定义别的 iterator 对象, 所以该方法就是返回该类内部具体的 迭代对象attribute

### 5.5.1. Generator Types



# 6. Python 流程控制

控制程序的执行顺序

- 运行逻辑控制
- 异常处理

## 6.1. 逻辑流程

- 逻辑流程控制
  - `break`和`continue` 同C语言是一样的
  - Python中, while 和for 后也可以紧跟着一个 else 代码块
    - 当循环条件为 False 跳出循环时, 程序会最先执行 else 代码块中的代码
    - 但是使用 `break` 跳出当前循环体之后, 该循环后的 `else` 代码块**也不会被执行**

### 6.1.1. `if` 表达式

1. `if` `elif`  `else` 是三个关键字, 后面接表达式和 `:`
2. 代码块记得加缩进
3. `pass` 是 Python 中的关键字, 程序需要占一个位置, 或者放一条语句, 但又不希望这条语句做任何事

```python
if 表达式:
  pass #内容
elif 表达式:
  pass
else:
  pass
```

### 6.1.2. while 表达式

内容同样不需要括号

```py
while 条件表达式：
    代码块

else:
    pass
```

### 6.1.3. for 表达式

1. for 循环中经常使用 `range()` 函数来指定循环
2. 在使用 for 循环遍历字典时, 经常会用到和字典相关的 3 个方法, 即 items()、keys() 以及 values()
   - 如果使用 for 循环直接遍历字典, 和遍历字典 keys() 方法的返回值是相同的

```python
for 迭代变量 in 字符串|列表|元组|字典|集合：
    代码块

my_dic = {'python教程':"http://c.biancheng.net/python/",\
         'shell教程':"http://c.biancheng.net/shell/",\
         'java教程':"http://c.biancheng.net/java/"}
for ele in my_dic.items():
   print('ele =', ele)
```

### 6.1.4. 列表推导式（又称列表解析式）

推导式（又称解析器）, 是 Python 独有的一种特性。  
使用推导式可以快速生成列表、元组、字典以及集合类型的数据,因此推导式又可细分为:  

- 列表推导式  用 `[]` 括起来
- 元组推导式  `()`
- 字典推导式  `{}`    表达式以键值对（key：value）的形式
- 集合推导式  `{}`    表达式不是键值对

- 可以这样认为, 它只是对 for 循环语句的格式做了一下简单的变形, 并用对应的符号括起来而已
- 可以在列表中放入任意类型的对象  返回结果将是一个新的列表  
- 可以包含多个循环的 for 表达式, 左侧是最外层, 右侧是最内层

```py
# [if 条件表达式] 不是必须的, 可以使用, 也可以省略
# 添加 if 条件语句, 这样列表推导式将只迭代那些符合条件的元素
[表达式 for 迭代变量 in 可迭代对象 [if 条件表达式] ]
(表达式 for 迭代变量 in 可迭代对象 [if 条件表达式] )
 {表达式 for 迭代变量 in 可迭代对象 [if 条件表达式]}

b_list = [x * x for x in a_range if x % 2 == 0]


test=[x*y for x in range(1,5) if x > 2 for y in range(1,4) if y < 3]
for x in range(1,5):
  if x > 2:
    for y in range(1,4):
      if y < 3:
        x*y
```

## 6.2. 异常流程控制

- python 的核心异常处理机制即 try except
- 异常处理中也可以使用 else
  - 代表程序不出现异常时执行的代码
  - 即不对应任何一种 exception 的情况
- 异常处理中还有一个 `finally` 块
  - 该块和 try 对应, 即不在乎程序中是否有 except 或者 else 块
  - 和else的逻辑相比, finally 不管是否发生异常最终都会被执行, 而 else 和 except 都是可能会被执行
  - 通常用于垃圾回收, 关闭文件, 关闭数据库连接
  - 甚至除了无视异常, 被 break 或者 return 语句推出的时候, finally 也会执行
    - finally 块中的 return 因此会覆盖 其他块中的 return, 因此不要在 finally 中使用 return
    - 唯一的例外是 python 解释器的退出语句 `os.exit(1)`

```py
try:
    可能产生异常的代码块
except [ (Error1, Error2, ... ) [as e] ]:
    处理异常的代码块1
except [ (Error3, Error4, ... ) [as e] ]:
    处理异常的代码块2
except  [Exception]:
    处理其它异常
else:
    无异常的情况
finally:
    最终的垃圾回收
```

- 语法格式
  - `Error1, Error2` 等代表具体的异常类型, 一个 except 块可以处理多种异常
  - `as e` 给异常一个别名, 方便调用异常
  - `except  [Exception]` 代表程序可能发生的所有异常情况, 通常放在最后兜底
    - 这种语句写在最后, 就算不加 Exception 也代表接受所有异常情况

### 6.2.1. 异常类

- 异常作为一个类也拥有对应的属性
  - args  记录了异常的错误编号和描述字符串
- python 预定义了许多异常类  
  - 所有的类都继承于 BaseException
  - 但是程序中的异常都继承于子类 Exception
  - 用户自定义的异常也应该继承 Exception

BaseException  
 +-- SystemExit  
 +-- KeyboardInterrupt  
 +-- GeneratorExit  
 +-- Exception  

```py
try:
    1/0
except Exception as e:
    # 访问异常的错误编号和详细信息
    print(e.args)
    print(str(e))
    print(repr(e))
# 输出
# ('division by zero',)
# division by zero
# ZeroDivisionError(division by zero',)
```

### 6.2.2. raise 语句

- raise 语句用于主动调取一个异常
- 语法格式: `raise [exceptionName [(reason)]]`
  - 可以指定异常类型, 默认是返回上文中已经捕获的异常, 否则返回 RuntimeError 异常
  - reason, 异常的描述信息

```py
raise
# RuntimeError: No active exception to reraise

raise ZeroDivisionError
# ZeroDivisionError

raise ZeroDivisionError("除数不能为零")
# ZeroDivisionError: 除数不能为零
```

### 6.2.3. assert 语句

类似于 C 语言的 assert

- 判断某个表达式的值, 如果值为真, 则程序可以继续往下执行
- 反之, Python 解释器会报 AssertionError 错误。

不能滥用 assert, 很多情况下, 程序中出现的不同情况都是意料之中的, 需要用不同的方案去处理  
有时用条件语句进行判断更为合适, 而对于程序中可能出现的一些异常, 要记得用 try except 语句处理  

```py
assert 表达式

# 等同于

if 表达式==True:
    程序继续执行
else:
    程序报 AssertionError 错误
```

### 6.2.4. 异常信息捕获

或许异常的详细信息

- 使用 sys 中的 exc_info 方法
  - 返回当前的异常信息, 以元组返回, 有三个元素
    - type: 异常类型的名称, 即异常类的名字
    - value: 异常实例
    - traceback: traceback的对象, 需要调用 traceback 包才能进行解析
- 使用 traceback 模块

```py
try:
  1/0
except:
  # 输出异常信息
  print(sys.exc_info())

  # 使用模块来打印 traceback
  traceback.print_tb(sys.exc_info()[2])
# (<class 'ZeroDivisionError'>, ZeroDivisionError('division by zero',), <traceback object at 0x000001FCF638DD48>)

# File "C:\Users\mengma\Desktop\demo.py", line 7, in <module>
```

### 6.2.5. 自定义异常类

- 自定义的异常类通常继承自 Exception 类, 名字以 `Error` 结尾
- 自定义异常类也是一个类, 而且只能被 raise 调用, 不会被解释器触发
  - 实现一些必要的类方法有助于异常类的使用

```py
class SelfExceptionError(Exception):
    def __init__(self,value):
      self.value=value
    def __str__(self):
      return ("{} is invalid input".format(repr(self.value)))

try:
    raise SelfExceptionError(1)
except SelfExceptionError as err:
    print('error: {}'.format(err))

# error: 1 is invalid input
```

# 7. python 的函数

函数的定义

```py
def 函数名(参数列表):
   """
   三重双引号的长字符串用来表示函数的文档
   """
    pass
    
    # 可以不需要返回值
    # [return [返回值]]
```

- 原理上来说 python的所有函数都需要返回值,

- 不手动写出的话会在幕后给没有 return 的函数添加 return None

## 7.1. 函数参数

**Python的值传递和引用传递**

1. Python没有引用传递
2. 如果参数是可变对象(列表,字典) 则传入的是对象的地址, 从结果上造成了引用传递
3. 如果需要让函数修改某些数据, 则可以通过把这些数据包装成列表、字典等可变对象, 然后把列表、字典等可变对象作为参数传入函数

**参数的识别位置**

1. 位置识别,  也称必备参数
   - 大体上同C语言一样, 具有顺序, 且必须传入
2. 关键字参数
   - 关键字参数必须放在顺序参数后面, 此时不需要传入顺序
   - 根据`调用时`加在参数前面的关键字来识别,通常用于带默认值的可选参数
   - `形参名=默认值` 的方式即可创建默认参数

- 例如`print()` 函数的`end`和`sep`用来指定参数末尾打印什么,参数之间打印什么
- `print('Hellow',end='')` 在打印后不会自动换行
- `print('a','b','c')` 在参数之间会默认隔一个空格
- `print('a','b','c',seq=',')` 会输出 **a,b,c**

### 7.1.1. Python的可变参数

- 可变参数, 即允许定义参数个数可变的函数。这样当调用该函数时, 可以向其传入任意多个参数, 包括不传参数。  
- `*args` 可变参数的值默认是空元组
- `**kwargs` 可变参数的值默认是空字典
- 一般情况下会把可变参数放在最后一个
  - 如果把可变参数放在了前面, 那么所有参数都会成为可变参数的一个元素
  - 需要将剩下的普通参数用关键字指定

```py
# 定义可变参数: 形参前添加一个 '*'
# * 传入的参数会被包装成一个  元组 
def dis_str(home, *str) :
   for s in str :
      print(s)

# 定义可变参数：形参前添加两个'*'
# **kwargs 表示创建一个名为 kwargs 的空字典, 该字典可以接收任意多个以关键字参数赋值的实际参数

def dis_str(*str,**course) :
    print(str)
    print(course)

#调用函数
dis_str("http://c.biancheng.net",\
        "http://c.biancheng.net/python/",\
        shell教程="http://c.biancheng.net/shell/",\
        go教程="http://c.biancheng.net/golang/",\
        java教程="http://c.biancheng.net/java/")


```

### 7.1.2. 逆向参数收集

- 通过星符号, 直接将  "列表、元组" 和 "字典" 作为函数参数
- Python 会将其进行拆分, 把其中存储的元素按照次序分给函数中的各个形参。

```py
def dis_str(name,add) :
    print("name:",name)
    print("add",add)


data = ["Python教程","http://c.biancheng.net/python/"]
#使用逆向参数元组收集方式传值
dis_str(*data)

data = {'name':"Python教程",'add':"http://c.biancheng.net/python/"}
#使用逆向参数字典收集方式传值
dis_str(**data)

```

### 7.1.3. partial 偏函数

- partial关键字, 位于 `functools` 模块中
- 是专门用于对函数进行二次封装的功能 : 定义偏函数
- `偏函数名 = partial(func, *args, **kwargs)`

所谓偏函数, 就是给部分参数预先绑定为指定值, 从而得到一个新的函数.  
和一般的函数默认参数相比, 偏函数直接就隐藏了部分的参数, 减少了可变参数的数目,  
而且可以在原本的单一函数上引申出复数个不同的函数, 使得调用更加简单  

```py
from functools import partial
#定义个原函数
def display(name,age):
    print("name:",name,"age:",age)
#定义偏函数，其封装了 display() 函数，并为 name 参数设置了默认参数
GaryFun = partial(display,name = 'Gary')
#由于 name 参数已经有默认值，因此调用偏函数时，可以不指定
GaryFun(age = 13)
```

### 7.1.4. 闭包函数

同偏函数一样, 闭包函数也是削减参数个数的一种函数封装方法,  
闭包函数的定义有特殊的语法:

- 在函数体中, 返回的不是一个具体的值, 而是一个函数
- 给闭包函数传入一个自由变量, 获得一个自由变量确定的另一个函数

从原理上来讲:

- 对于闭包函数, 传入的自由变量会以另一种方式被记录
- 闭包函数比普通函数多一个 `__closure__` 属性, 记录着自由变量的地址
- `__closure__` 是一个元组, 代表可以记录多个自由变量

```py
#闭包函数，其中 exponent 称为自由变量
def nth_power(exponent):
    # 函数中定义另一个函数, 并固定了自由变量
    def exponent_of(base):
        return base ** exponent
    # 返回值是 exponent_of 函数
    return exponent_of 

# 通过闭包函数, 获得了计算一个数的平方的函数
square = nth_power(2) 

# 计算平方
print(square(2))

# 查看 __closure__ 的值
print(square.__closure__)
```

## 7.2. 函数的文档

通过在合理的地方放置多行注释, python 可以方便的将其作为函数说明文档输出

- 函数的说明文档通常位于函数内部、所有代码的最前面
- 调用 help() 函数
- 调用 函数的内置属性 `__doc__`

```py
def str_max(str1,str2):
    '''
    比较 2 个字符串的大小
    '''
    str = str1 if str1 > str2 else str2
    return str

# 两种方法输出函数说明文档
help(str_max)
print(str_max.__doc__)

```

## 7.3. yield 表达式

- yield 用来定义一个生成器函数或者异步生成器函数中
- 因此只能被用在函数体的定义里, 使得该函数不再是一个普通函数
  - def 函数生成一个生成器
  - async def 函数生成一个异步生成器

- 作用
  - 提高函数的复用性, 如果需要一个数列, 最好不要直接打印出数列, 而是返回一个 List
  - 但是返回数列会导致, 函数在运行中占用的内存会随着返回的 list 的长度而增大
  - 如果要控制内存占用, 最好不要用 List, 来保存中间结果, 而是通过 iterable 对象来迭代

- 使用方法
  - 把 yield 替换掉 return 即可
  - 下次再调用该函数的时候会从上次 yield 的地方继续运行
$\rightarrow$
```py
def gen():  # defines a generator function
    yield 123

async def agen(): # defines an asynchronous generator function
    yield 123


# range(1000) 函数会返回一个 1000 的list
for i in range(1000): pass
# xrange 不会返回一个一整个 list, 而是返回一个 iterable 对象, 在每次迭代中返回下一个数值, 内存空间占用很小
for i in xrange(1000): pass


# 使用 yield 编写的斐波那契生成数列, 效果不变, 内存只占用常数项
def fab(max): 
    n, a, b = 0, 0, 1 
    while n < max: 
        yield b      # 使用 yield
        # print b 
        a, b = b, a + b 
        n = n + 1
 
for n in fab(5): 
    print n

# 使用类编写相应的 iterate 方法也能有同样的效果, 但是代码不简洁
class Fab(object): 
  def __init__(self, max): 
      self.max = max 
      self.n, self.a, self.b = 0, 0, 1 

  def __iter__(self): 
      return self 

  def next(self): 
      if self.n < self.max: 
          r = self.b 
          self.a, self.b = self.b, self.a + self.b 
          self.n = self.n + 1 
          return r 
      raise StopIteration()

# 使用 yield 编写定长缓冲区读取文件

def read_file(fpath): 
    BLOCK_SIZE = 1024 
    with open(fpath, 'rb') as f: 
        while True: 
            block = f.read(BLOCK_SIZE) 
            if block: 
                yield block 
            else: 
                return
                
```

## 7.4. lambda 表达式 匿名函数

- lambda 表达式, 又称匿名函数, 常用来表示内部仅包含 1 行表达式的函数.  
- 如果一个函数的函数体仅有 1 行表达式, 则该函数就可以用 lambda 表达式来代替。  
- 优点:
  - 代码更加简洁
  - 可以在局部域中定义, 使用完后立即释放, 提高程序的性能

语法:
`name = lambda [list] : 表达式`  

- `lambda` 是保留字
- `list` 作为可选参数, 等同于定义函数指定的参数列表
- `name`   该表达式的名称
等同于:

```py
def name(list):
    return 表达式
name(list)
```

## 7.5. 函数的异常处理

如果 `try` 子句中的代码发生了错误, 则程序立即到 `except` 中的代码去执行  

因此将`try` 放到函数中和直接放到代码段中会有不同的效果, 会影响程序执行的流程, 因此一般**将异常封装在函数里**.

```python
try:
   return 42/eggs
except ZeroDivisionError:
   print('divide zero')
#对于try-except模块,可以使用else来使得程序在正确运行时进入下一模块
else:
```

python有很多的error类:  
`ZeroDivisionError`  
`ValueError` 对于输入数据的类型不符
`FileNotFoundError` 打开文件的路径不对, 文件不存在  

# 8. python 的类

类同C++无太大区别  

- python可以动态的给类添加变量和方法
  - 添加使用正常赋值
  - 删除使用 `del` 保留字

## 8.1. 定义

python 的类通过`class`定义 , python的 `类名` 一般以大写字母开头的驼峰式  

```py
class TheFirstDemo:
    '''这是一个学习Python定义的第一个类'''

    # 定义类的构造函数  
    def __init__(self,<other parameter>):
        pass

    # 下面定义了一个类属性
    add = 'http://c.biancheng.net'

    # 下面定义了一个say方法
    def say(self, content):
        print(content)


```

1. 同理在定义下方第一行可以写类的文档
2. `__init__` 是保留的构造函数名, 可以不写构造函数, 会隐式的定义默认的构造函数
3. `self` 是保留的参数, 类的所有方法都需要有`self`参数, 但是该参数不需要真的传参

## 8.2. self

同C++一样, 指向方法的调用者  

- self 只是约定俗成, 实际上只要该参数在第一个位置, 名字随意
- 动态给对象添加的方法不能自动绑定 `self` 参数, 需要传参, 即第一个参数需要是对象本身

```py
# 先定义一个函数
def info(self):
    print("---info函数---", self)
# 使用info对clanguage的foo方法赋值（动态绑定方法）
clanguage.foo = info

# 调用的时候需要传参
clanguage.foo(clanguage)

# 使用lambda表达式为clanguage对象的bar方法赋值（动态绑定方法）
clanguage.bar = lambda self: print('--lambda表达式--', self)
# 同理调用的时候需要传参
clanguage.bar(clanguage)
```

## 8.3. 类的变量

1. 类变量: 类变量指的是在类中, 但在各个类方法外定义的变量。
   - 所有类的实例化对象都同时共享类变量, 即在所有实例化对象中是作为公用资源存在的
   - 既可以使用类名直接调用, 也可以使用类的实例化对象调用
2. 实例变量: 在任意类方法内部, 以`self.变量名`的方式定义的变量
   - 只作用于调用方法的对象。
   - 只能通过对象名访问, 无法通过类名访问。
3. 局部变量 : 类方法中普通方法定义, 不使用 `self.` 来定义的变量
   - 函数执行完成后, 局部变量也会被销毁。

## 8.4. 类方法 静态方法

1. `@classmethod` 修饰的方法为类方法
   - 相当于C++的类的静态方法
2. `@staticmethod` 修饰的方法为静态方法
   - 相当于在类中定义了一个普通函数, 但是属于类的命名空间
   - 没有 self 参数, 因此不会绑定类对象, 也因此不能调用任何类对象和类方法
3. 不用任何修改的方法为实例方法
4. `@classmethod` 和 `@staticmethod` 都是函数装饰器

- 实例方法
  - 通常通过对象访问, 最少包含一个 `self` 参数
  - 通过类名访问, 需要提供对象参数 `CLanguage.say(clang)`
  - 用类的实例对象访问类成员的方式称为绑定方法, 而用类名调用类成员的方式称为非绑定方法。

- 在实际编程中, 几乎不会用到类方法和静态方法
- 特殊的场景中（例如工厂模式中）, 使用类方法和静态方法也是很不错的选择。

```py
class CLanguage:
    #类构造方法, 也属于实例方法
    def __init__(self):
        self.name = "C语言中文网"
        self.add = "http://c.biancheng.net"

    # 类方法需要使用＠classmethod修饰符进行修饰
    #下面定义了一个类方法
    @classmethod
    def info(cls):
        print("正在调用类方法",cls)

    @staticmethod
    def infos(name,add):
        print(name,add)


# 类方法也要包含一个参数, 通常将其命名为 cls
# Python 会自动将类本身绑定给 cls 参数

# 类方法推荐使用类名直接调用, 当然也可以使用实例对象来调用（不推荐）
CLanguage.info()

# 类的静态方法中无法调用任何类属性和类方法, 所以能用静态方法完成的工作都可以用普通函数完成
CLanguage.infos("C语言中文网","http://c.biancheng.net")

```

- 用类的实例对象访问类成员的方式称为绑定方法

- 而用类名调用类成员的方式称为非绑定方法。

## 8.5. 类的描述符

- 通过使用描述符, 可以让程序员在引用一个对象属性时自定义要完成的工作
- 一个类可以将属性管理全权委托给描述符类
- 描述符是 Python 中复杂属性访问的基础

描述符类基于以下 3 个特殊方法,  即这三个最基础特殊方法构成了描述符协议

1. `__set__(self, obj, type=None)` ：在设置属性时将调用这一方法
2. `__get__(self, obj, value)`     ：在读取属性时将调用这一方法
3. `__delete__(self, obj)`         ：对属性调用 del 时将调用这一方法。

- 实现了 set 和 get 描述符方法的描述符类被称为 `数据描述符`
- 只实现了 get 被称为 `非数据描述符`

在通过类查找属性时, 即 `类对象.属性` 的调用方法时, 都会隐式的调用 `__getattribute__()` 这一特殊方法  
它会按照下列顺序查找该属性：  

1. 验证该属性是否为类实例对象的数据描述符；
2. 如果不是, 就查看该属性是否能在类实例对象的 `__dict__` 中找到；
3. 最后, 查看该属性是否为类实例对象的非数据描述符。

```py
#描述符类
class revealAccess:
    def __init__(self, initval = None, name = 'var'):
        self.val = initval
        self.name = name
    def __get__(self, obj, objtype):
        print("Retrieving",self.name)
        return self.val
    def __set__(self, obj, val):
        print("updating",self.name)
        self.val = val
class myClass:
    x = revealAccess(10,'var "x"')
    y = 5
```

## 8.6. 类的封装

- Python 并没有提供 public、private 这些修饰符
    默认情况下, Python 类中的变量和方法都是公有（public）的, 它们的名称前都没有下划线（_）；
- 如果类中的变量和函数, 其名称以`双下划线`“__”开头, 则该变量（函数）为私有变量（私有函数）, 其属性等同于 private。
  - python 的私有变量的封装方法不是结构性的, 只是在内部将私有变量的名称更改了
  - 对于一个变量 `__私有变量` 在执行过程中实际的变量名变成了 `_类名__私有变量` 因此仍然可以在外部访问
- 用 `类对象.属性` 的方法访问类中的属性是不妥的, 破坏了类的封装性

### 8.6.1. property()

- 为了实现类似于C++的类私有变量, 即只能通过类方法来间接操作类属性, 一般都会设置 getter setter 方法
- 虽然保护了封装性, 但是调用起来非常麻烦
- property 就是解决上述问题的

property函数可以设置任意前几个参数  
类属性可以按照原本的 `类对象.属性` 的方法访问, 但是实际执行的时候是通过对应的类方法

```py
# 属性名=property(fget=None, fset=None, fdel=None, doc=None)
class CLanguage:
    #构造函数
    def __init__(self,n):
        self.__name = n
    #设置 name 属性值的函数
    def setname(self,n):
        self.__name = n
    #访问nema属性值的函数
    def getname(self):
        return self.__name
    #删除name属性值的函数
    def delname(self):
        self.__name="xxx"
    #为name 属性配置 property() 函数
    name = property(getname, setname, delname, '指明出处')
```

### 8.6.2. @property 装饰器

- 同 property 的作用一样, 这个装饰器的目的也是一样, 方便调用代码的书写
- 通过该装饰器可以让方法的调用变得和属性一样 - 即不带括号
- 同 `@property` 相同系列的还有
  - `@方法名.setter`
  - `@方法名.deleter`

  -

```py
class Rect:
    def __init__(self,area):
        self.__area = area
    @property
    def area(self):
        return self.__area

    @area.setter
    def area(self, value):
        self.__area = value

    @area.deleter
    def area(self):
        self.__area = 0
rect = Rect(30)
#直接通过方法名来访问 area 方法
print("矩形的面积是：",rect.area)    

del rect.area
print("删除后的area值为：",rect.area)
```

## 8.7. 类的继承和多态

- 类的继承, 在定义子类的时候, 将父类放在子类之后的圆括号即可
  - `class 类名(父类1, 父类2, ...)：`
  - 所有类都隐式继承自 `object` 类
  - python 支持多继承, 大部分的对象语言都不允许多继承
    - 对于多个父类中的同名方法, 以最早出现的父类为准

### 8.7.1. super
  
- 子类如果定义了自己的构造方法, 则里面必须要调用父类的构造方法
- 在子类中的构造方法中, 调用父类构造方法的方式有 2 种, 分别是：
  - 类可以看做一个独立空间, 在类的外部调用其中的实例方法, 可以向调用普通函数那样, 只不过需要额外备注类名（此方式又称为未绑定方法）；
  - 使用 super() 函数。但如果涉及多继承, 该函数只能调用`第一个直接父类的构造方法`。
  - 及到多继承时, 在子类构造函数中, 调用第一个父类构造方法的方式有以上 2 种, 而调用其它父类构造方法的方式只能使用未绑定方法。

`super().__init__(self,...)`

### 8.7.2. MRO Method Resolution Order

# 9. python 的模块和包

Python 的核心封装功能

## 9.1. 导入模块或包

导入语句有多种写法

1. `import <name1> [as <别名>], <name2> [as <别名>]`
   - 来导入一个包,可以使用其中的函数  
   - 在使用的时候要加上包名或者别名 `name.function()`  
2. `from <name> import fun1[as <别名>], fun2[as <别名>]`
   - 只导入指定函数,这时包中函数不再需要包名
   - 直接使用成员名（或别名）即可

模块导入的查找顺序

- 在当前目录, 即当前执行的程序文件所在目录下查找；
- 到 PYTHONPATH（环境变量）下的每个目录中查找；
- 到 Python 默认的安装目录下查找。

以上所有涉及到的目录, 都保存在标准模块 `sys` 的 `sys.path` 变量  
因此在自定义包中的 `__init__` 文件中, 都会通过该变量进行路径添加  

```py
import sys, os
import warnings
if(not(os.path.dirname(os.path.realpath(__file__)) in sys.path)):
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
```

## 9.2. 自定义模块

- 只要是 Python 程序, 都可以作为模块导入
- 模块名就是文件名, 不带`.py`

自定义模块的文档  

- 同理, 在模块开头的位置用多行字符串定义
- 会自动赋值给该模块的 `__doc__` 变量

模块的自定义导入

1. 名称以下划线（单下划线“_”或者双下划线“__”）开头的变量、函数和类不会被导入, 属于本地
2. 该模块设有 `__all__` 变量时, `from 模块名 import *` 只能导入该变量指定的成员, 用其他方式导入则不受影响

**自定义模块的运行**

导入后, 默认会执行包中的全部代码

- 通常情况下, 为了检验模块中代码的正确性, 往往需要在模块中为其设计一段测试代码
- 为让导入该模块的代码不自动执行测试代码, 借助`内置变量   __name__`

1. 当直接运行一个模块时, name 变量的值为 `__main__`
2. 将模块被导入其他程序中并运行该程序时, 处于模块中的 `__name__` 变量的值就变成了模块名

```py
# __name__  变量
if __name__ == '__main__':
  say() # 执行测试代码
```

## 9.3. 包

包就是文件夹, 只不过在该文件夹下必须存在一个名为 `__init__.py` 的文件

- 每个包的目录下都必须建立一个 `__init__.py` 的模块, 可以是一个空模块, 也可以是初始化代码
  - `__init__.py` 不同于其他模块文件, 此模块的模块名不是 `__init__` , 而是它所在的包名, 即文件夹名
- 文件夹的名称就是包名

包导入的语法和导入模块相同, 只不过多了 `.` 点号, 用指定导入层级

**__init__ 的编写**

1. 导入包就等同于导入该包中的 `__init__.py` 文件
2. 该文件的主要作用是导入该包内的其他模块

```py
# 在多文件编程中, 通过编写 __init__ 快速导入自定义包

# 导入当前文件夹下的模块
# 不同的书写方法会导致导入包后函数的调用

# 需要用 包名.模块名.
from . import module1

# 虽然功能定义在模块2里, 但是调用时只用 包名.函数名
from .module2 import * 
```

## 9.4. 包信息调取

任何模块或者包作为一段python代码, 有自己内部定义的变量和类

- help() 函数可以获取传入的对象的信息
- `__doc__` 变量可以获取用户自己书写的文档
- `__file__` 变量可以获取当前模块的源文件系统位置
  - 包的话就是 `__init__.py` 文件的路径
  - 模块就是源文件的路径

# 10. Python 的文件操作

- python 的文件操作核心就是 `file object`

- 与 C++ 的文件对象一样, 也有地点指针
  - `.tell()` 返回一个整数, 代表当前的文件指针位置, 即从文件开始的bytes数或者字符数
  - `.seek(offset,whence)`
    - offset 表示偏移量
    - whence表示偏移起始位置
    - 0表示从文件开始
    - 1表示从当前位置
    - 2表示从文件末尾

## 10.1. open 打开文件

- 打开一个文件路径并返回 file object
- 如果不能打开, 会 raise OSError
  - 需要配合 with 语句来确保打开范围
  - 如果不用 with 的话, 需要手动调用 f.close()来确保文件正常关闭

```py
open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None)
# file    ：一个 path-like object
# mode    : 一个字符串用来指定打开的模式, 默认是'r', 

```

| Character | Meaning                                                         |
| --------- | --------------------------------------------------------------- |
| 'r'       | open for reading                                                |
| 'w'       | open for writing, truncating the file first                     |
| 'x'       | open for exclusive creation, failing if the file already exists |
| 'a'       | open for writing, appending to the end of the file if it exists |
| 'b'       | binary mode                                                     |
| 't'       | text mode (default)                                             |
| '+'       | open for updating (reading and writing)                         |

## 10.2. 读取文件

- 在通过 open 得到 file object 后, 即可通过相关方法来读取文件
  - .read(size) 读取一定数量的数据
    - size的单位是字符数或者bytes数
    - 没有 size 默认读取整个文件
  - .readline() 读取单个行
    - 这种方法会读取到行末的 `\n`
    - 使用 for 循环来遍历 file object 时也是默认以行为单位
  - 希望将数据转换成 list
    - list(f) 直接转化
    - f.readlines() 返回一个list

```python
#这个文件对象只在with的Block里面有效
with open('pi_digits.txt') as file_object:
  # 1. read() 读取整个文件
  contents = file_object.read()         
  
  # 2. 按行来读取文件
  for line in file_object:
    # 行末已有换行符
    print(line,end='')

  # 3. 将文件读取到一个list里
  lines = file_object.readlines()
for line in lines:  #可以在block外读取文件内容
  print(line.rstrip()) # 可以用rstrip方法来清除文件的换行符
```

## 10.3. 写入文件

- 要想写入文件,需要在文件对象创建的时候指定`'w'`参数,或者`'a'`参数
- 使用方法 `.write(str)` 传入要写入的字符串, 同时会返回写入的字符个数

```python
# 使用 write()方法来写入内容
# 记得自己输入换行符
file_object.write("I like programming.\n") 
```

## 10.4. 结构化读取文件

使用 enumerate 可以按行获取文件内容  
`for j, data in enumerate(openfile)`

```py
with open('animal.txt', 'r') as openfile:
  for j, data in enumerate(openfile):
    if j % n == 0:
      print(f'Line number {str(j)}. Content: {data}')
```

# 11. 正则表达式 re包

要在python中使用正则表达式, 需要导入`re`包  
`import  re`  

官方文档[https://docs.python.org/3/library/re.html]

## 11.1. 使用正则表达式的基础函数

```python
# 基础search函数,用来查找表达式第一次出现的位置
m = re.search(regex, text)
print(m)   # 查找的结果是这样的一个体 <_sre.SRE_Match object; span=(6, 8), match='in'>
print(m.span()) # (6, 8) 返回一个元组, 包含了两个值, 分别是匹配的开始位置索引和结束位置索引
print(m.group()) # 'abc'  直接返回匹配的字符串内容

# 与search不同, findall返回一个列表, 表项就是匹配的字符串内容, 若无匹配的内容则返回空列表
re.findall(regex, text)

# re.split使用正则表达式的分割字符串, 返回分割后的字符串列表  
x = re.split(' ', text)

# re.sub用来进行字符串替换  
x = re.sub('被替换的字符串', '填充的字符串', text)

```

## 11.2. 正则表达式-单字符

```python
# 使用[]来代表一个单字符的可选列表
# 匹配sta/stb/stc
re.search('st[abc]', 'understand')

# 使用破折号代表范围
re.search('st[0-9a-fA-F]', 'best_first99')

# 在[]中使用^来表示不包含的符号
# 匹配一个非数字的字符
re.search('[^0-9]', 'str1ke_one')

# 使用'\' 来进行关键字转义
# 查找 e-o
re.search('e\-o', 'str1ke-one')

# 万能字符 . , '.'表示任意一个单字符
re.search('.e', 'str1ke^one')

# 字符集合
# \w is equal to [a-zA-Z0-9_]
# \W is equivalent to [^a-zA-Z0-9_].
# \d is equal to [0-9].
# \D is equal to [^0-9]
# \s matches any whitespace character (including tab and newline). 
# \S is the opposite of  \s  It matches any character that isn't whitespace.


# 位置字符
# 使用^或\A来代表字符串起始位置
re.search('^fi', 'first')

# 使用$或\Z来代表字符串末尾
re.search('st$', 'first')

# \b 来同时表示以上两种, 代表边界位置
# \B does the opposite of, 表示处于字符串中间
re.search(r'\bst', 'first strike')
re.search(r'\Bst\B', 'strike first estimate')
```

## 11.3. 正则表达式-多次匹配

```python
#  * 代表任意次  + 代表至少1次 ,? 代表0或者1次
# * 和 + 和 ?都是默认贪心, 查找出现最多的字符串段 
# *? 和 +? ??来指定多字符匹配会选择重复次数最少的片段


# 查找纯数字片段
re.search('[0-9]*', '1st_strike')

# 使用{} 来直接指定前一个字符的匹配重复次数
re.search('x\d{3}x', 'x789x')

# 使用{m,n} 来指定重复次数区间
# 同样默认是贪心匹配, 使用{m,n}? 来代表最小匹配
# {,} 省略m或者n的数字来代表最小0次和最大无限次

re.search('x-{2,4}x', 'x--xx----x')
re.search('x-{2,4}?x', 'x--xx----x')

# 使用()来直接指定匹配的字符串片段,多字符匹配
re.search('(bar)+', 'foo barbarbar baz')

```

# 12. Python 的包 环境管理

[官方推荐文档]<https://packaging.python.org/guides/tool-recommendations/>  


## 12.1. Python的包管理

包管理工具有很多
* [pip](./pip.md)

### 12.1.1. distutils 和 setuptools

distutils是 python 标准库的一部分  
用于方便的打包和安装模块  是常用的 setup.py 的实现模块  
setuptools 是对 distutils 的增强, 尤其是引入了包依赖管理  

[distutils打包文档]<https://docs.python.org/3.8/distutils/>  

使用方法: 在工作目录下, 要打包的脚本文件是`foo.py` 和 `bar.py`  
创建 `setup.py`  

```py
from distutils.core import setup  # 导入打包函数
setup(
    name='fooBar',  # 包名 
    version='1.0',  # 版本
    author='Will',  # 作者
    author_email='wilber@sh.com',   # 邮件地址
    url='http://www.cnblogs.com/wilber2013/',   # 作者网址
    py_modules=['foo', 'bar'],      # 要打包的文件名称
)
```

然后在工作目录下运行  `python setup.py sdist`  则会打包好  `fooBar-1.0.zip`  
安装者则需要解压压缩包  然后运行  `python setup.py install`  就可以使用包了


## 12.2. 环境管理

在开发Python应用程序的时候, 系统安装的Python3只有一个版本：3.4。所有第三方的包都会被pip安装到Python3的site-packages目录下。

如果我们要同时开发多个应用程序, 那这些应用程序都会共用一个Python, 就是安装在系统的Python 3  
如果应用A需要jinja 2.7, 而应用B需要jinja 2.6怎么办  

有多种虚拟环境配置方法  

### 12.2.1. venv  

Source code: Lib/venv/  
一般已经安装在了较新的 python 版本中了  因为是从 3.3 版本开始自带的, 这个工具也仅仅支持 python 3.3 和以后版本  
创建一个轻量级虚拟环境, 与系统的运行环境相独立, 有自己的 Python Binary  

使用 `venv` 命令进行虚拟环境操作  

```shell
# vene ENV_DIR
python3 -m venv tutorial-env
python3 -m venv /path/to/new/virtual/environment


# 激活虚拟环境
source tutorial-env/bin/activate

```

### 12.2.2. virtualenv

virtualenv 是目前最流行的 python 虚拟环境配置工具

- 同时支持 python2 和 python3
- 可以为每个虚拟环境指定 python 解释器 并选择不继承基础版本的包。

使用pip3安装  
`pip3 install virtualenv`  

```shell
# 测试安装 查看版本
virtualenv --version

# 创建虚拟环境
cd my_project
virtualenv my_project_env

# 指定python 执行器
-p /usr/bin/python2.7

# 激活虚拟环境
source my_project_env/bin/activate
# 停用 回到系统默认的Python解释器
deactivate
```

### 12.2.3. virtualenvwrapper

`pip install virtualenv virtualenvwrapper`  

virtualenvwrapper 是对 virtualenv 的一个封装, 目的是使后者更好用
使用 shell 脚本开发, 不支持 Windows  

它使得和虚拟环境工作变得愉快许多

- 将您的所有虚拟环境在一个地方。
- 包装用于管理虚拟环境（创建, 删除, 复制）。
- 使用一个命令来环境之间进行切换。

```shell


#设置环境变量 这样所有的虚拟环境都默认保存到这个目录
export WORKON_HOME=~/Envs  
#创建虚拟环境管理目录
mkdir -p $WORKON_HOME


# 每次要想使用virtualenvwrapper 工具时, 都必须先激活virtualenvwrapper.sh
find / -name virtualenvwrapper.sh #找到virtualenvwrapper.sh的路径
source 路径 #激活virtualenvwrapper.sh

# 创建虚拟环境  
# 该工具是统一在当前用户的envs文件夹下创建, 并且会自动进入到该虚拟环境下  
mkvirtualenv ENV
mkvirtualenv ENV  --python=python2.7

# 进入虚拟环境目录  
cdvirtualenv

Create an environment with `mkvirtualenv`

Activate an environment (or switch to a different one) with `workon`

Deactivate an environment with` deactivate`

Remove an environment with`rmvirtualenv`

# 在当前文件夹创建独立运行环境-命名
# 得到独立第三方包的环境, 并且指定解释器是python3
$ mkvirtualenv cv -p python3

# 进入虚拟环境  
source venv/bin/activate  

#接下来就可以在该虚拟环境下pip安装包或者做各种事了, 比如要安装django2.0版本就可以：
pip install django==2.0

```

**其他命令**:

- workon `ENV`          : 启用虚拟环境
- deactivate            : 停止虚拟环境
- rmvirtualenv `ENV`    : 删除一个虚拟环境
- lsvirtualenv          : 列举所有环境
- cdvirtualenv          : 导航到当前激活的虚拟环境的目录中, 比如说这样您就能够浏览它的 site-packages
- cdsitepackages        : 和上面的类似, 但是是直接进入到 site-packages 目录中
- lssitepackages        : 显示 site-packages 目录中的内容


# python 解释器

此节用于学习 python CLI 以及解释器的各种环境配置

## python Environment variables

会对 python 解释器起作用的环境变量  


## 

好像一般的包都不能够通过终端直接访问

`python -m pip --version`

查看一个包的版本  
