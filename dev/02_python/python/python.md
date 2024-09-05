- [1. The Python Language Reference](#1-the-python-language-reference)
  - [1.1. 版本区别  2和3](#11-版本区别--2和3)
  - [1.2. Python PEP文档](#12-python-pep文档)
  - [1.3. 底层语言](#13-底层语言)
- [2. Python的语法](#2-python的语法)
  - [2.1. Python 书写规范 (PEP 8)](#21-python-书写规范-pep-8)
  - [2.2. Python 保留字](#22-python-保留字)
  - [2.3. python 的类型提示](#23-python-的类型提示)
  - [2.4. 作用域](#24-作用域)
- [3. Python的操作符与变量](#3-python的操作符与变量)
  - [3.1. 基础类型](#31-基础类型)
  - [3.2. 转义字符](#32-转义字符)
  - [3.3. 运算符](#33-运算符)
- [4. Python 流程控制](#4-python-流程控制)
  - [4.1. 逻辑流程](#41-逻辑流程)
    - [4.1.1. for 表达式](#411-for-表达式)
    - [4.1.2. 列表推导式（又称列表解析式）](#412-列表推导式又称列表解析式)
  - [4.2. 异常流程控制](#42-异常流程控制)
    - [4.2.1. 异常类](#421-异常类)
    - [4.2.2. raise 语句](#422-raise-语句)
    - [4.2.3. assert 语句](#423-assert-语句)
    - [4.2.4. 异常信息捕获](#424-异常信息捕获)
    - [4.2.5. 自定义异常类](#425-自定义异常类)
- [5. python 的函数](#5-python-的函数)
  - [5.1. 函数参数](#51-函数参数)
    - [5.1.1. Python的可变参数](#511-python的可变参数)
    - [5.1.2. 逆向参数收集](#512-逆向参数收集)
    - [5.1.3. partial 偏函数](#513-partial-偏函数)
    - [5.1.4. 闭包函数](#514-闭包函数)
  - [5.2. 函数的文档](#52-函数的文档)
  - [5.3. yield 表达式](#53-yield-表达式)
  - [5.4. lambda 表达式 匿名函数](#54-lambda-表达式-匿名函数)
  - [5.5. 函数的异常处理](#55-函数的异常处理)
- [6. Python 的文件操作](#6-python-的文件操作)
  - [6.1. open 打开文件](#61-open-打开文件)
  - [6.2. 读取文件](#62-读取文件)
  - [6.3. 写入文件](#63-写入文件)
  - [6.4. 结构化读取文件](#64-结构化读取文件)
- [7. Command line and environment](#7-command-line-and-environment)
  - [7.1. Command line](#71-command-line)
    - [7.1.1. -m module-name](#711--m-module-name)
    - [7.1.2. 杂项选项](#712-杂项选项)
  - [7.2. python Environment variables](#72-python-environment-variables)
- [8. Lexical analysis - 词法分析](#8-lexical-analysis---词法分析)
  - [8.1. Identifiers and keywords - 标识符和关键字](#81-identifiers-and-keywords---标识符和关键字)
    - [8.1.1. Keywords](#811-keywords)
    - [8.1.2. Soft Keywords](#812-soft-keywords)
    - [8.1.3. Reserved classes of identifiers - 保留的类](#813-reserved-classes-of-identifiers---保留的类)
- [9. Data model - 数据模型](#9-data-model---数据模型)
  - [9.1. Objects, values and types - 对象, 值, 类型 的定义](#91-objects-values-and-types---对象-值-类型-的定义)
  - [9.2. The standard type hierarchy](#92-the-standard-type-hierarchy)
  - [9.3. Special method names - 特殊方法名称](#93-special-method-names---特殊方法名称)
  - [Coroutines - 协程](#coroutines---协程)
- [10. Execution model](#10-execution-model)
- [11. The import system - 模组导入系统](#11-the-import-system---模组导入系统)
  - [11.1. 导入模块或包](#111-导入模块或包)
  - [11.2. 自定义模块](#112-自定义模块)
  - [11.3. 包](#113-包)
  - [11.4. 包信息调取](#114-包信息调取)
- [12. Expressions 表达式](#12-expressions-表达式)
  - [Arithmetic conversions - 基础数值类型转换规则](#arithmetic-conversions---基础数值类型转换规则)
- [13. Simple statements - 简单语句](#13-simple-statements---简单语句)
- [14. Compound Statements - 复合语句](#14-compound-statements---复合语句)
  - [14.1. The if statement](#141-the-if-statement)
  - [14.2. The while statement](#142-the-while-statement)
  - [The with statement - with 表达式](#the-with-statement---with-表达式)
  - [14.3. The match statement - python3.10 加入](#143-the-match-statement---python310-加入)
  - [Function definitions - 函数的定义](#function-definitions---函数的定义)
  - [14.4. Class definitions - 类的定义](#144-class-definitions---类的定义)
  - [Coroutines - 协程函数的定义](#coroutines---协程函数的定义)
  - [Type parameter lists](#type-parameter-lists)
  - [14.6. self](#146-self)
  - [14.7. 类的变量](#147-类的变量)
  - [14.8. 类方法 静态方法](#148-类方法-静态方法)
  - [14.9. 类的描述符](#149-类的描述符)
  - [14.10. 类的封装](#1410-类的封装)
    - [14.10.1. property()](#14101-property)
    - [14.10.2. @property 装饰器](#14102-property-装饰器)
  - [14.11. 类的继承和多态](#1411-类的继承和多态)
    - [14.11.1. super](#14111-super)
    - [14.11.2. MRO Method Resolution Order](#14112-mro-method-resolution-order)
- [15. Top-level components - 顶层复合语句](#15-top-level-components---顶层复合语句)

# 1. The Python Language Reference

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



# 3. Python的操作符与变量

Python是弱类型语言

- 变量无须声明就可以直接赋值
- 变量的数据类型可以随时改变
- 使用 type() 内置函数类检测某个变量或者表达式的类型

## 3.1. 基础类型

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

## 3.2. 转义字符

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

## 3.3. 运算符

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



# 4. Python 流程控制

控制程序的执行顺序

- 运行逻辑控制
- 异常处理

## 4.1. 逻辑流程

- 逻辑流程控制
  - `break`和`continue` 同C语言是一样的
  - Python中, while 和for 后也可以紧跟着一个 else 代码块
    - 当循环条件为 False 跳出循环时, 程序会最先执行 else 代码块中的代码
    - 但是使用 `break` 跳出当前循环体之后, 该循环后的 `else` 代码块**也不会被执行**


### 4.1.1. for 表达式

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

### 4.1.2. 列表推导式（又称列表解析式）

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

## 4.2. 异常流程控制

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

### 4.2.1. 异常类

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

### 4.2.2. raise 语句

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

### 4.2.3. assert 语句

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

### 4.2.4. 异常信息捕获

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

### 4.2.5. 自定义异常类

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

# 5. python 的函数

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

## 5.1. 函数参数

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

### 5.1.1. Python的可变参数

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

### 5.1.2. 逆向参数收集

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

### 5.1.3. partial 偏函数

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

### 5.1.4. 闭包函数

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

## 5.2. 函数的文档

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

## 5.3. yield 表达式

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

## 5.4. lambda 表达式 匿名函数

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

## 5.5. 函数的异常处理

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

# 6. Python 的文件操作

- python 的文件操作核心就是 `file object`

- 与 C++ 的文件对象一样, 也有地点指针
  - `.tell()` 返回一个整数, 代表当前的文件指针位置, 即从文件开始的bytes数或者字符数
  - `.seek(offset,whence)`
    - offset 表示偏移量
    - whence表示偏移起始位置
    - 0表示从文件开始
    - 1表示从当前位置
    - 2表示从文件末尾

## 6.1. open 打开文件

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

## 6.2. 读取文件

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

## 6.3. 写入文件

- 要想写入文件,需要在文件对象创建的时候指定`'w'`参数,或者`'a'`参数
- 使用方法 `.write(str)` 传入要写入的字符串, 同时会返回写入的字符个数

```python
# 使用 write()方法来写入内容
# 记得自己输入换行符
file_object.write("I like programming.\n") 
```

## 6.4. 结构化读取文件

使用 enumerate 可以按行获取文件内容  
`for j, data in enumerate(openfile)`

```py
with open('animal.txt', 'r') as openfile:
  for j, data in enumerate(openfile):
    if j % n == 0:
      print(f'Line number {str(j)}. Content: {data}')
```


# 7. Command line and environment

此节用于学习 python CLI 以及解释器的各种环境配置

## 7.1. Command line

使用 python CLI 的形式
* `python myscript.py` 简单运行一个脚本
* `python [-bBdEhiIOqsSuvVWx?] [-c command | -m module-name | script | - ] [args]`  完整的 CLI 可能输入

### 7.1.1. -m module-name

Search `sys.path` for the named module and execute its contents as the __main__ module.

执行一个 包自己的 `main` 函数 `<pkg>.__main__`, module-name 不需要带上 `.py` 后缀  


### 7.1.2. 杂项选项

`python -m pip --version`



## 7.2. python Environment variables

会对 python 解释器起作用的环境变量  

# 8. Lexical analysis - 词法分析

就是编译原理的那个词法分析: 由于 python 是解释性语言, 所以代码会直接经过词法分析生成 token 传入执行设备
python 默认使用 Unicode utf-8 来解码代码, 可以通过 encoding declaration 来更改


## 8.1. Identifiers and keywords - 标识符和关键字


### 8.1.1. Keywords

以下的字符完全作为 python 语言的关键字 (与内置函数的含义不同), 是构成语法的关键, 使用时要完全正确拼写

```
False      await      else       import     pass
None       break      except     in         raise
True       class      finally    is         return
and        continue   for        lambda     try
as         def        from       nonlocal   while
assert     del        global     not        with
async      elif       if         or         yield
```
### 8.1.2. Soft Keywords 

仅在特定的上下文 context 下被解释为关键字, 是 3.10 后加入的新的语法的功能  包括 

`match, case, type, _`  4 种, soft keyward 的区分是发生在 parser level 
* match, case, and _ are used in the `match` statement
* type is used in the `type` statement.


### 8.1.3. Reserved classes of identifiers - 保留的类

某些 identifiers 会带有特殊含义, 通过特殊的前缀来实现区别

# 9. Data model - 数据模型

## 9.1. Objects, values and types - 对象, 值, 类型 的定义
<!-- 完 -->

objects: 是 python 对于数据的抽象表示, python 里面所有的数据都是 对象, 或者用对象之间的关系表示
* 代码也由对象表示
* 所有对象都有 identity, type, value, 其中 identity 是所有对象的唯一标识符, 类似于对象在内存中的地址, 但不一定是真的地址
  * 可以通过 `is` 来进行对象的身份比较
  * `id()` 返回一个表示对象身份的整数
  * 对于 CPython 来说, `id(x)` 就是内存地址

type: 决定了该对象所支持的操作, 例如能否获取长度, 以及该对象可能存入的值的类型
* 对于一个对象来说, type 是不可更改的
* type 本身也是一个对象


value: 可以改变 value 的对象称为 mutable, 不可改变 value 的对象称为 immutable
* 如果 immutable 里包含了 mutable, 那么 mutable 的值改变的时候， immutable 的实际上的值也发生了改变
* immutable 只是从形式上不允许任何值操作


Python 的 Object 永远没有显示的删除功能, 如果一个 object 不可被访问 (unreachable), 那么它可能会被当作垃圾收集
* 从思想上 reachable object 永远不会被收集
* 如果满足了上述条件, 则垃圾收集就是 Python 语言的实现质量的问题
* CPython 的垃圾收集实现:
  * 名为 reference-counting scheme with (optional) delayed detection of cyclically linked garbage 的方案
  * 即当一个 object 不在被 reference, 即变得 unreachable, 则该对象理论上会被立即收集
    * but is not guaranteed to collect garbage containing circular references
    * 什么是 circular references? 两个结构体, 内部互相引用彼此, python 的 `gc` 包里面提供了 circular references 的检测接口, 用于获取内存中unreachable 的  circular references
  * 就算被收集, 不代表对象会被立即摧毁, 因此在 Python 中永远应该显式的关闭文件 `file.close()`

设计到非正常垃圾收集的语句有:
* debugging facilities 会导致 object 持续保持活跃
* try except : 语句很有可能会使得 try 中的对象在语句结束后仍然活跃

因此凡是涉及到外部资源, 例如文件读取的操作的时候, python 极其推荐
* 使用 try ... finally 来确保终结文件的打开状态
* 使用 with 来方便的实现 上下文管理



一个对象是否是 immutable 是十分重要的, 因为这涉及到在实现的过程中是否进行对象复用
* 例如, int 类型本身是不可变的类型, 值1 和值2 是两个分别的对象 (只是举个例子, 不代表真实实现)
* 执行 `a=1, b=1` 的时候, 有可能会使得 a,b 的值都是 id 相同的一个对象, 因为 1 值的对象可以复用
* 因此在使用 object identity 的特性进行编程的时候需要格外谨慎
* 注意:
  * `a=[] , b=[]` 在 python 中会确保引用两个不同的, 唯一的, 新创建的空列表
  * `a=b=[]` 会保证 a,b 引用相同的 空列表



## 9.2. The standard type hierarchy

## 9.3. Special method names - 特殊方法名称

一个 class 可以通过实现特殊名称的 方法, 来实现由特殊语法调用的某些操作. 即运算符重载, 索引, 切片 等

例如 `__getitem__()` 方法可以让类能够进行索引操作  

同时, 如果运算符并未有对应操作的对应方法, 则会唤起 ERROR, 通常是  AttributeError or TypeError

手动将某个特殊名称的方法设置为 None, 表示对应类不可以进行相应操作, 可以避免默认行为导致的歧义

## Coroutines - 协程




# 10. Execution model

# 11. The import system - 模组导入系统



Python 的核心封装功能

## 11.1. 导入模块或包

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

## 11.2. 自定义模块

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

## 11.3. 包

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

## 11.4. 包信息调取

任何模块或者包作为一段python代码, 有自己内部定义的变量和类

- help() 函数可以获取传入的对象的信息
- `__doc__` 变量可以获取用户自己书写的文档
- `__file__` 变量可以获取当前模块的源文件系统位置
  - 包的话就是 `__init__.py` 文件的路径
  - 模块就是源文件的路径

# 12. Expressions 表达式

该章节说明了 meaning of the elements of expressions

## Arithmetic conversions - 基础数值类型转换规则 

对于应用运算符的两个数值类型
* 若有一方是复数, 则另一方转为复数
* 若有一方是浮点数, 则另一方转为浮点数
* 否则, 两边都必须是整数



# 13. Simple statements - 简单语句

# 14. Compound Statements - 复合语句

复合语句 (包含了其他语法的代码组), 通常复合语句都会有多行  

## 14.1. The if statement

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

## 14.2. The while statement

内容同样不需要括号

```py
while 条件表达式：
    代码块

else:
    pass
```

## The with statement - with 表达式

通过 context manager 的方式来包装块的执行

```py
with_stmt          ::=  "with" ( "(" with_stmt_contents ","? ")" | with_stmt_contents ) ":" suite
with_stmt_contents ::=  with_item ("," with_item)*
with_item          ::=  expression ["as" target]
```

自底向上说明该语法:
* `with_item` : context expression 会被执行, 其结果会作为 context manager, `["as" target]` 用于给 context manager 赋予名称


## 14.3. The match statement - python3.10 加入

在 python 3.10 以前的版本使用该语句会报错, 向下兼容性很差


## Function definitions - 函数的定义

## 14.4. Class definitions - 类的定义
<!-- 完 -->

类同C++无太大区别

在 python 中, 类似乎更倾向于高级的数据结构, 因为 python 的很多基础的数据类型都类似于一个类, 可以参照 标准数据类型层级的说明: https://docs.python.org/3/reference/datamodel.html#types

python 格式的定义语法为

```py
classdef    ::=  [decorators] "class" classname [type_params] [inheritance] ":" suite
inheritance ::=  "(" [argument_list] ")"
classname   ::=  identifier
```

`inheritance` : 类的继承列表
* 该列表中的类会作为基类, 用于定义该新的用户类
* 该列表中的 所有类都需要被允许进行子类的定义 
  * each item in the list should evaluate to a class object which allows subclassing
* 如果没有定义的类作为继承对象, 则会激动继承 python 的底层基类 `object`

```py
class Foo:
    pass
# 等同于
class Foo(object):
    pass

# 1. 同理在定义下方第一行可以写类的文档
# 2. `__init__` 是保留的构造函数名, 可以不写构造函数, 会隐式的定义默认的构造函数
# 3. `self` 是保留的参数, 类的所有方法都需要有`self`参数, 但是该参数不需要真的传参
```

* `suite` : 在定义 class 之后, 会在 new execution frame 中执行 suite 里的语句
  * 使用 newly created local namespace 以及 original global namespace
  * suite 执行完成后, 对应的 execution frame 会被丢弃, 而 class 的 local namespace 会保留
* 接下来, 会按照继承列表 创建基类, 并将对应的 local namespace 存入 attribute dictionary
* 在原本的 origin local namespace 里将 class name 绑定到该 class object


* `decorators` : 类也同函数一样, 可以被修饰
  * (python3.9) class 可以被任何有效的  `assignment_expression` 修饰. 在 3.9 之前, 该语法更加严格
* `type parameters` : 在 class's name 之后, 可以接续一个由方括号括起来的 `type parameters`
  * (python3.12) type parameters 是 3.12 的新功能, 参照后面的章节
  * 这回向 静态类型检查器(static type checkers) 表明该类是 generic 的
  * 在 runtime 中, 对应的 type params 可以从  `__type_params__` 的属性中提取


类的属性:
* 在 class definition 中, 定义的变量 (不带 self.) 为 class attributes, 是 instances 共享的内容
* instance attributes 是每个实例独有的, 要定义需要使用 `self.name = value`
* class/instance attributes 在访问的时候使用相同的方法  `self.name`, 且名字相同的时候 instance 会覆盖 class attributes
  * class attributes 在某种程度上可以作为属性的默认值
  * 但是如何 class attributes 使用了 mutable values, 修改某一个实例会导致其他所有实例的值都更改导致非预期的行为, 因此不推荐
* 




## Coroutines - 协程函数的定义


## Type parameter lists 

Python 3.12 新功能



## 14.6. self

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

## 14.7. 类的变量

1. 类变量: 类变量指的是在类中, 但在各个类方法外定义的变量。
   - 所有类的实例化对象都同时共享类变量, 即在所有实例化对象中是作为公用资源存在的
   - 既可以使用类名直接调用, 也可以使用类的实例化对象调用
2. 实例变量: 在任意类方法内部, 以`self.变量名`的方式定义的变量
   - 只作用于调用方法的对象。
   - 只能通过对象名访问, 无法通过类名访问。
3. 局部变量 : 类方法中普通方法定义, 不使用 `self.` 来定义的变量
   - 函数执行完成后, 局部变量也会被销毁。

## 14.8. 类方法 静态方法

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

## 14.9. 类的描述符

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

## 14.10. 类的封装

- Python 并没有提供 public、private 这些修饰符
    默认情况下, Python 类中的变量和方法都是公有（public）的, 它们的名称前都没有下划线（_）；
- 如果类中的变量和函数, 其名称以`双下划线`“__”开头, 则该变量（函数）为私有变量（私有函数）, 其属性等同于 private。
  - python 的私有变量的封装方法不是结构性的, 只是在内部将私有变量的名称更改了
  - 对于一个变量 `__私有变量` 在执行过程中实际的变量名变成了 `_类名__私有变量` 因此仍然可以在外部访问
- 用 `类对象.属性` 的方法访问类中的属性是不妥的, 破坏了类的封装性

### 14.10.1. property()

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

### 14.10.2. @property 装饰器

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

## 14.11. 类的继承和多态

- 类的继承, 在定义子类的时候, 将父类放在子类之后的圆括号即可
  - `class 类名(父类1, 父类2, ...)：`
  - 所有类都隐式继承自 `object` 类
  - python 支持多继承, 大部分的对象语言都不允许多继承
    - 对于多个父类中的同名方法, 以最早出现的父类为准

### 14.11.1. super
  
- 子类如果定义了自己的构造方法, 则里面必须要调用父类的构造方法
- 在子类中的构造方法中, 调用父类构造方法的方式有 2 种, 分别是：
  - 类可以看做一个独立空间, 在类的外部调用其中的实例方法, 可以向调用普通函数那样, 只不过需要额外备注类名（此方式又称为未绑定方法）；
  - 使用 super() 函数。但如果涉及多继承, 该函数只能调用`第一个直接父类的构造方法`。
  - 及到多继承时, 在子类构造函数中, 调用第一个父类构造方法的方式有以上 2 种, 而调用其它父类构造方法的方式只能使用未绑定方法。

`super().__init__(self,...)`

### 14.11.2. MRO Method Resolution Order


# 15. Top-level components - 顶层复合语句
