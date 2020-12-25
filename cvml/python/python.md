# 1. Python 的背景

**优点**:
1. 语法简单
   1. Python 不要求在每个语句的最后写分号，当然写上也没错；
   2. 定义变量时不需要指明类型，甚至可以给同一个变量赋值不同类型的数据。
   3. 这两点也是 PHP、JavaScript、MATLAB 等常见脚本语言都具备的特性。
2. Python 是开源的
3. Python 是免费的
4. Python 是高级语言
   1. Python 封装较深，屏蔽了很多底层细节
   2. Python 会自动管理内存（需要时自动分配，不需要时自动释放）
5. 解释型语言，能跨平台
6. 是面向对象的编程语言
7. Python 功能强大（模块众多）
8. 可扩展性强
   1. 类库的底层代码不一定都是 Python，还有很多 C/C++ 的身影。
   2. 当需要一段关键代码运行速度更快时，就可以使用 C/C++ 语言实现，然后在 Python 中调用它们
   3. 依靠其良好的扩展性弥补了运行效率慢的缺点

**缺点**:
1. 运行速度慢是解释型语言的通病
   1. Python 的运行速度几乎是最慢的，不但远远慢于 C/C++，还慢于 Java
2. 代码加密困难

当代应用场景速度不是第一要求
1. 多花钱就可以堆出高性能的硬件，硬件性能的提升可以弥补软件性能的不足
2. 比如网站，用户打开一个网页的大部分时间是在等待网络请求(500ms) ，而不是等待服务器执行网页程序(20ms)

## 1.1. 版本区别  2和3

1. Python 3.x print函数代替了print语句
   *  Python2.x 中，输出数据使用的是 Print 语句 `print "3,4"`
   *  Python 3.x 中，print 语句没有了，取而代之的是 print 函数 `print(3,4)`
2. Python 3.x 默认使用 UTF-8 编码
   * Python 2.x 默认采用的 ASCII 编码
   * Python 3.x 默认使用 UTF-8 编码
3. Python 3.x 除法运算
   * 使用运算符 `/` 除法运算，结果也会是浮点数
   * 运算符 `//` 进行的除法运算叫做 floor 除法（向下取整）。
4. Python 3.x 八进制字面量表示
   * 在 Python 2.x 版本中，所有类型的对象都是直接被抛出的
   * 只有继承 BaseException 的对象才可以被抛出
5. Python 3.x 不等于运算符
   * 在 Python 3.x 中去掉了 <>，只有 ！=

## 1.2. Python PEP文档

PEP（Python Enhancement Proposal），全称是 Python 改进方案  

有以下 3 个用途：

    通知：汇总 Python 核心开发者重要的信息，并通过 Python 发布日程；
    标准化：提供代码风格、文档或者其他指导意见；
    设计：对提交的功能进行说明。

## 1.3. 底层语言

1. Python，又称为 CPython。平时我们所讨论的 Python，指的其实就是 CPython ，是用 C 语言编写实现的
2. 用 Java 语言实现的 Python 称为 JPython
3. 用 .net 实现的 Python 称为 IronPython 
4. PyPy 中，Python 解释器本身是用 Python 编写的

# 2. Python的语法

1. 标识符的命名和C没有区别, 但是以下划线开头的标识符有特殊含义
   * 单下划线开头的标识符`_width`, 表示不能直接访问的类属性，其无法通过 `from...import* `的方式导入
   * 双下划线开头的标识符`__add`, 表示类的私有成员
   * 双下划线作为开头和结尾的标识符`__init__`, 是专用标识符
2. python 的字符串可以用单引号`'ABC'`**或者**双引号表示`"ABC"`
3. `#` 号代表单行注释, `""" 注释 """ ` 三个双引号中间的内容代表多行注释 (其实是长字符串) 
4. python代码第一行的声明 需要注明解释器脚本, 这样可以在命令行中直接调用脚本
  * windows 下 `#! python3`
  * OS X 下, `#! /usr/bin/env python3`
  * Linux 下, `#! /usr/bin/python`    
5. python 使用缩进和冒号`:` 来分隔代码块
   * Python 中实现对代码的缩进，可以使用空格或者 Tab 键实现。
   * 同一个级别代码块的**缩进量**必须一样，否则解释器会报 SyntaxError 异常错误
   * 具体缩进量为多少，并不做硬性规定
   * Tab 键, 通常情况下都是采用 4 个空格长度作为一个缩进量
   * （默认情况下，一个 Tab 键就表示 4 个空格）
   * 缩进规则在大多数情况下有用,可以使用 `/` 来使得一条指令可以跨越到下一行,同时使下一行的缩进无效,增加代码可读性  

* 程序运行中使用<kbd>Ctrl</kbd>+<kbd>C</kbd> 可以立即终止程序



## 2.1. Python 书写规范 (PEP 8)

1. 每个 import 语句只导入一个模块，尽量避免一个语句导入多个模块
2. 每行代码不需要分号, 也不要用分号将两条命令放在同一行
3. 每行不超过 80 个字符，如果超过，建议使用小括号将多行内容隐式的连接起来，而不推荐使用反斜杠 `\` 进行连接

**必要的空行可以增加代码的可读**
1. 顶级定义（如函数或类的定义）之间空两行
2. 而方法定义之间空一行
3. 运算符两侧、函数参数之间以及逗号两侧，都建议使用空格进行分隔

## 2.2. 常用的最基础的函数
* `myname=input()` 接受键盘输入的一个字符串,结果存储到变量`myname`
* `len()`  括号中传入一个字符串
* 类型转换函数
  * `str()`  括号中传入整型数字,返回字符串
  * `int()`  `float()` 分别为传入数字的字符串,并将类型转换为数字
* 使用命令行参数  命令行参数存储在 `sys.argv` 中,以列表的形式,第一个项是文件名,从第二项开始是第一个命令行参数

## 2.3. Python 保留字

Python 包含的保留字可以执行命令进行查看
```py
import keyword
keyword.kwlist
```

True False  : bool类型的值  

and  
as  
assert  
break  
class  
continue  
def
del  
elif  
else  
except  
finally  
for
from  
global  
if  
import  
in
is  
lambda  
nonlocal  
not  
None  
or
pass  
raise  
return  
try  
while
with  
yield 	  

# 3. 内置函数

可以直接使用, 不需要导入某个模块, 解释器自带的函数叫做内置函数  

    在 Python 2.x 中，print 是一个关键字；到了 Python 3.x 中，print 变成了内置函数。

不要使用内置函数的名字作为标识符使用, 虽然这样做 Python 解释器不会报错, 但这会导致同名的内置函数被覆盖，从而无法使用  

* type() : 输出变量或者表达式的类型


* ord() : 字符转化成 Unicode 码
* chr() : 数字的Unicode 码 转化成字符

* id()  : 获取变量或者对象的内存地址

其他内置函数
```
abs() 	delattr() 	hash() 	memoryview() 	set()
all() 	dict() 	help() 	min() 	setattr()
any() 	dir() 	hex() 	next() 	slicea()
ascii() 	divmod()  	object() 	sorted()
bin() 	enumerate() 	oct() 	staticmethod()
 	eval() 	 	open() 	str()
breakpoint() 	exec() 	isinstance()  	sum()
bytearray() 	filter() 	issubclass() 	pow() 	super()
bytes() 	 	iter() 	tuple()
callable() 	format() 	len() 	property() 	type()
frozenset() 	list() 	range() 	vars()
classmethod() 	getattr() 	locals() 	repr() 	zip()
compile() 	globals() 	map() 	reversed() 	__import__()
complex() 	hasattr() 	max() 	round()
```

## 3.1. print()

`print (value,...,sep=' ',end='\n',file=sys.stdout,flush=False)`  

* value 参数可以接受任意多个变量或值，因此 print() 函数可以输出多个值
* `end` 参数的默认值是 `\n`
* `sep` 参数的默认值是 ` ` 空格
* `file` 参数的默认值为 sys.stdout, 代表标准输出, 即屏幕输出  
* `flush` 参数用于控制输出缓存，该参数一般保持为 `False` 即可，这样可以获得较好的性能


```py
print("读者名：",user_name,"年龄：",user_age)

#打开文件
f = open("demo.txt","w")
print('123456',file=f)
```

### 3.1.1. 格式化输出

类似于C语言的格式化输出, print() 函数提供了类似的功能

* 在 print() 函数中，由引号包围的是格式化字符串，它相当于一个字符串模板
* 格式化字符串中的`%` 是转换说明符（Conversion Specifier）只是一个占位符
* 转换说明符会被后面表达式替代
* `%` 也被用作分隔符, 格式化字符串后接一个`%` 再接输出的表达式
* 格式化字符串中可以包含多个转换说明符, 这个时候也得提供多个表达式, 此时多个表达式必须使用小括号( )包围起来

```py
name = "C语言中文网"
age = 8
url = "http://c.biancheng.net/"
print("%s已经%d岁了，它的网址是%s。" % (name, age, url))
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


### 控制输出方式

转换说明符的`%`和类型字符中间可以加入控制内容
* 数字指定最小输出宽度  `%10d`
  * 当数据的实际宽度小于指定宽度时，会在左侧以空格补齐
  * 当数据的实际宽度大于指定宽度时，会按照数据的实际宽度输出, 不会裁剪
* 对齐标志指定对齐方法, 加在`数字之前`, 默认是左侧补空格, 数据靠右边
  * `-` 指定左对齐, 
  * `+` 指定数字数据带正负号 `+-`, 只对数字数据生效
  * `0` 指定数据用 0 代替空格补足, 只对数字数据生效
    * 左对齐时因为右侧补零会改变整数的数值, 因此左对齐时不对整数生效
* 对小数指定精度, 放在`最小宽度数据之后, 用点号隔开`  
  * `%m.nf` `m` 表示最小宽度，`n` 表示输出精度，`.`是必须存在的, f表示小数输出


```py
n = 1234567
print("%10d." % n)
print("%5d." % n)

f = 3.141592653
# 最小宽度为8，小数点后保留3位
print("%8.3f" % f)
# 最小宽度为8，小数点后保留3位，左边补0
print("%08.3f" % f)
# 最小宽度为8，小数点后保留3位，左边补0，带符号
print("%+08.3f" % f)


```

## 3.2. input()
字符串形式接受用户输入  

`str = input(tipmsg)`  
* str   : 表示输入存入的变量
* tipmsg: 表示在控制台中输出的提示信息, 提示输入什么内容

## 3.3. 类型转换函数

因为 input() 输入的内容都被识别为字符串, 因此需要类型转换
* int()
* bool()
* float()

# 4. Python的操作符与变量


Python是弱类型语言
* 变量无须声明就可以直接赋值
* 变量的数据类型可以随时改变
* 使用 type() 内置函数类检测某个变量或者表达式的类型

## 4.1. 基础类型

整数, 浮点数, 复数, 字符串, bytes, 布尔    

* 整数不分类型, Python 整数的取值范围是无限的
  * 当所用数值超过计算机自身的计算能力时，Python 会自动转用高精度计算（大数计算）
  * 为了提高数字的可读性, 可以用下划线来分割数字而不影响本身的值 `1_000`
  * 可以使用多种进制表示数字
    * 十进制形式的整数不能以 0 作为开头，除非这个数值本身就是 0
    * 二进制形式书写时以`0b`或`0B`开头
    * 八进制形式以`0o`或`0O`开头
    * 十六进制形式书写时以`0x`或`0X`开头
* 浮点数也只有一种类型
  * 书写小数时必须包含一个小数点，否则会被 Python 当作整数处理`34.6`
  * 用指数形式生成的值也是小数 `2e2`= 200
* Python 语言本身就支持复数
  * 复数的虚部以j或者J作为后缀 `a + bj`
  * 本身支持简单的复数运算
* 字符串必须由双引号`" "`或者单引号`' '`包围
  * 字符串中的引号可以在引号前面添加反斜杠`\`就可以对引号进行转义
  * 普通字符串或者长字符串的开头加上`r`前缀，就变成了原始字符串
  * 如果字符串内容中出现了单引号，那么我们可以使用双引号包围字符串，反之亦然
  * 使用三个单引号或者双引号其实是 Python 长字符串的写法
    * 所谓长字符串，就是可以直接换行, 可以在字符串中放置任何内容，包括单引号和双引号
    * 长字符串中的换行、空格、缩进等空白符都会原样输出
* bytes 类型用来表示一个二进制序列, 类似于无类型的指针
  * 只负责以字节序列的形式（二进制形式）来存储数据，至于这些数据到底表示什么内容, 完全由程序的解析方式决定
  * bytes 只是简单地记录内存中的原始数据，至于如何使用这些数据，bytes 并不在意
  * bytes 是一个类，调用它的构造方法，也就是 bytes(), 可以将指定内容转化成字节串
  * 如果字符串的内容都是 ASCII 字符，那么直接在字符串前面添加b前缀就可以将字符串转换成 bytes
* 字符串有一个 encode()  将字符串转换成bytes()
* bytes 有一个 decode() 将字节串转换成字符串
* bool 类型用来表示真或者假, 真假都是python的保留字
  * True 
  * False 




## 4.2. 运算符

* `+ - * / % `与C语言相同
* `== != > < >= <=` 都与C语言相同
  * `==`和`!=`可以用于所有类型,若两边类型不同则永远不会相等
* **Boolean**型的值只能为 `True` 和 `False` 没有单引号,首字母必须大写
  * 逻辑运算与C语言的操作符不同 
  * && -> `and` , || -> `or` , ! -> `not`     顺序  not>and>or
  * 对于整型的`0`,浮点的`0.0`,字符串的 `''` 都被认为是`False` 其余为 `True` 
* `None` 是NoneType类型的唯一值,类似于C语言的 `null`
* `**` 代表求指数 2**8=256
* `//` 除法取商



### 4.2.1. `if` 表达式
```python
if 表达式:
  pass #内容
elif 表达式:
  pass
else:
  pass
```
### 4.2.2. `while` 表达式
内容同样不需要括号

`break`和`continue` 同C语言是一样的

### 4.2.3. `for` 表达式

```python
for target_list in expression_list:
  pass
```
for 循环中经常使用 `range()` 函数来指定循环
```python
for i in range(5)  #代表依次 i=0,1,2,3,4,
#range(<start>,<end>)
range(12,16)   #代表从12开始
#range(<start>,<end>,<stepWidth>)
range(12,16,2) #设置间隔，步长可以为负数
```
### 4.2.4. 列表推导式（又称列表解析式） 

它的结构是在一个中括号里包含一个表达式，然后是一个for语句，然后是 0 个或多个 for 或者 if 语句  
可以在列表中放入任意类型的对象  
返回结果将是一个新的列表  
在这个以 `if` 和 `for` 语句为上下文的表达式运行完成之后产生。    


执行顺序：各语句之间是嵌套关系，左边第二个语句是最外层，依次往右进一层，左边第一条语句是最后一层。

```py
test=[x*y for x in range(1,5) if x > 2 for y in range(1,4) if y < 3]

for x in range(1,5):
  if x > 2:
    for y in range(1,4):
      if y < 3:
        x*y
```


## 4.3. python的包的导入

使用 `import <name>` 来导入一个包,可以使用其中的函数  
在使用的时候要加上包名 `name.function()`  
使用 `from <name> import *` 可以只单独导入一个函数,这时包中函数不再需要包名   
使用 `as <alias>` 来为导入的包或者包中函数添加别名, 使得调用更方便  

### 4.3.1. 常用的包
* 包的下载 , 使用pip来下载及管理包
  * `pip install ModuleName` 下载一个包  
  * `pip install -u ModuleName` 下载一个已经安装包的更新版本  
* random
  * `random.randint(*,*)` 给出两个参数整数之间的一个随机整数
* sys
  * `sys.exit()` 提前结束程序


## 4.4. python的函数及作用域

### 4.4.1. 定义
```python
def functionName(): #python的函数参数不需要类型`
""" 关于函数的注释放在三个双引号中 """
  functionpass
  return None
  # python的所有函数都需要返回值,就算不手动写出也会在幕后给没有 return 的函数添加 return None
```
### 4.4.2. 参数
* 第一种是位置识别,同C语言一样
* 第二种是**关键字参数**,根据`调用时`加在参数前面的关键字来识别,通常用于可选参数
  * 例如`print()` 函数的`end`和`sep`用来指定参数末尾打印什么,参数之间打印什么
  * `print('Hellow',end='')` 在打印后不会自动换行
  * `print('a','b','c')` 在参数之间会默认隔一个空格
  * `print('a','b','c',seq=',')` 会输出 **a,b,c** 
  
### 4.4.3. 作用域

同C语言 全局和局部的覆盖原则也相同  
函数中引用外部变量 使用`global` 关键字
```python
spam=1
def fun()
    global spam
    spam=2
```
作用域规则
1. 如果一个变量在**所有函数**之外，它就是全局变量
2. 如果一个函数中
   1. 有针对变量的global语句，则他是全局变量  
   2. 否则，变量用于函数中的赋值语句，它就是局部变量
   3. 但是若该变量没有用在赋值语句，则仍然是全局变量
3. 在同一个函数中，同一个变量要么总是全局变量，要么总是局部变量，不能改变

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

## 4.5. 函数的异常处理

如果 `try` 子句中的代码发生了错误，则程序立即到 `except` 中的代码去执行  

因此将`try` 放到函数中和直接放到代码段中会有不同的效果，会影响程序执行的流程，因此一般**将异常封装在函数里**.

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

# 5. python 的列表和元组

## 5.1. 列表类型

* 列表值: 指的是列表本身,可以作为值保存在变量中,或者传递给函数
* 表项:   指的是列表内的值,用逗号分隔
* 使用下标访问列表中的每个值 `spam[0]` 若超出列表中值的个数,将返回 `IndexError`
* 列表中包含其他的列表值,构成多重列表,使用多重下标访问 `spam[0][1]`
* 负数下标: 使用负数来代表倒数的索引, `[-1]` 代表列表中倒数第一下标
* 切片下标: 从列表中取得多个值,结果是一个新的 `列表` 使用 `:` 来代表切片的两个整数 `spam[1:4]` ,从第一个下标开始到第二个下标,**不包括**第二个下标
* 下标的省略` [:2] [1:] [:}` 分别代表从头开始,到末尾,全部
* `len()`可以取得列表的长度
* `+` 操作可以连结两个列表
* `*` 给予一个整数,可以复制列表,效果同字符串
* ` del spam[2]` 删除列表中的下标值,后面所有值的下标都会往前移动一个
  
## 5.2. 列表的使用
python的列表总是动态的:
```python
catName=[]
while True:
    catName=catName+[name]
for name in catName:
    print(name)
```

### 5.2.1. 列表的循环
` for i in range(len(someList))`

### 5.2.2. 列表的查找
`in `和 `not in` 操作符  
可以确定一个值是否在列表中
```python
'howdy' in ['hello','hi','howdy']
返回值为 True 或者 False
```

### 5.2.3. 列表的赋值
使用列表可以同时为多个变量赋值  
`size,color,dispo=cat`  
变量的数目和列表的长度必须相等,否则报错  
`cat=['fat','black','loud']`

### 5.2.4. 增强的赋值操作
针对`+、-、*、/、% `操作符  
有`+=、-=、*=、/=、%=  `意义同C语言
```python
spam +=1 
spam=spam+1
```
`+= `可以完成字符串或列表的连接  
```python
spam='hello'  
spam+='world'
```
`*= `可以完成字符串或列表的复制  
```python
bacon=['zon']
bacon*=3
```

### 5.2.5. 列表类型的方法

* `index()` : `spam.index('hello')` 传入一个值,若该值存在于列表中,返回**第一次出现的下标**
* `append()` :`spam.append('hello') ` 将值添加到列表末尾
* `insert()`: `spam.insert(1,'chicken')` 将值添加到参数下标处
* `remove()`: `spam.remove('bat') ` 传入一个值，将该值从列表中删除，若该值不在列表中，返回`ValueError`错误,只会删除第一次出现的位置
  * 若知道删除的下标,就用 `del spam[2]` 在删除的同时返回删除的值,就用`a=spam.pop(2)`
  * 若知道要删除的值,就用`spam.remove('a')`
* `sort()`: 
  * `spam.sort()`数值的列表或者字符串的列表可以使用该方法
  * `spam.sort(reverse=True)` 指定反向排序
  * `sort()` 是当场排序,不是返回值,不能对混合值列表进行排序
  * `sort()` 是按照ASCII表排序,因此在大写是排在小写字母之前
    * 可以使用`spam.sort(key=str.lower)`使方法把所有的表项当作小写
* 函数`sorted()`
  * 可以返回一个排序的值,但是不会影响原本的值 `ab=sorted(spam)`

## 5.3. 字符串和元组
### 5.3.1. 字符串类型
字符串和元组都类似于列表
包括:  
* 下标取值
* 切片
* for循环遍历
* `len()`
* `in`和`not in`操作符  

列表是可变的,可以对值进行增删改,但字符串在定义后及不可改变,不可以使用列表的方法对字符串的一个字符重新赋值

改变一个字符串,可以使用切片+连接的方法  
`newName=name[:7]+'the'+name[8:]`

### 5.3.2. 元组数据类型 
元组与列表几乎一样,除了两点  
1. 元组输入时使用`()`而不是`[]` 例:`eggs=('hello',42,0.5)`
2. 元组属于不可变类型，同字符串一致,即值不能增改,字符串属于元组的一种

为了区分只有一个值的元组与与普通的值,在值的后面添加一个<kbd>,</kbd>
```python
>>>type ( ('hello',) )
<class 'tuple'>
>>>type( ('hello') )
<class 'str'>
```

元组与列表非常相似,因此有转换函数可以进行类型转换
* `list()` 和 `tuple()`
* `list('dog')` 结果输出 `['d','o','g']`

## 5.4. 关于引用
对于变量保存字符串和整数值，变量保存的是值本身  
对于列表，变量保存的是列表引用，类似于指针  

```python
spam=[1,2,3]
cheese=spam
```
1. 在函数调用时，对于列表参数是传递引用，因此需要注意可能的缺陷
2. copy模块,有时候为了不影响原来的列表或字典,使用copy模块,确保复制了列表
   * `spam=['A','B','C']`
   * `cheese=copy.copy(spam)`

# 6. 字典和结构化数据
## 6.1. 字典
同列表一样,字典是许多值的集合,但不同于列表的下标,字典的索引使用许多不同的数据类型,不只是整数

字典的定义使用花括号 `{}`
```python
myCat={ 'size':'fat','color':'gray','disposition':'loud' }
```
输入 `myCat['size']` 会输出`'fat'`
当然也可以使用数字作为索引,但是完全不受限制,不必从0开始

字典的内容完全不做排序,没有**第一个**的概念

创建空字典 `ice_cream_1 = {}`  
加入一个新的键值对: ` ice_cream_1['flavor'] = 'vanilla' `  
删除一个键值对 ` del ice_cream_1['flavor'] `  

## 6.2. 字典的使用
字典有三个打印方法
* `keys()`  :返回键值
* `values()`:具体值
* `items()` :键-值对
这些方法返回的不是真正的列表，不能被修改，不能使用 append() 方法，但可以用于 `for` 循环

可以使用 `list(spam.keys())` 将返回的值转换为列表

### 6.2.1. get() 方法
使用`spam.get('索引',备用值)`可以安全的通过索引访问一个值,当这个索引在字典中不存在时返回备用值  
若不使用`get`方法则会报错

### 6.2.2. setdefault() 方法
`setdefault('索引',值)` 可以安全的初始化一个键,当该键存在则返回键值,键不存在则设定为第二个参数

### 6.2.3. 字典的优化打印
自带的`print()`函数会把字典打印在一行  
使用`pprint`模块,可以优化打印
```python
someDictionay={}
pprint.pprint(someDictionary) # 可以每个键值对打印在一行
string=pprint.pformat(someDictionary) #返回一个和 上一行打印内容相同的字符串
print(string) # 效果和第一行相同
```

## 6.3. 集合的使用
集合（set）是一个无序的**不重复元素**序列。  
可以使用大括号` { } `或者` set() `函数创建集合  
注意: 创建一个空集合必须用` set()` 而不是 `{ }`，因为` { } `是用来创建一个空字典  

* 添加元素 `s.add( x )` 如果 x 已经存在于集合, 则无改变, 
  *  `thisset.add("Facebook")`
* 更高级的添加元素 `s.update( x ) ` 这里 x 可以可以是列表，元组，字典等, x 还可以有多个用逗号分开
  * `thisset.update([1,4],[5,6])  `
* 移除元素 `s.remove( x )` 如果元素不存在就会错误
* 另一个不会发生错误的移除元素方法 `s.discard( x )`
* 清空集合 `s.clear()`
* ` pop()`	随机移除元素

# 7. 字符串操作

## 7.1. 字符串的输入
* python的字符串使用 `''` 单引号输入  
* 也可以使用 `"" `双引号输入,区别在于使用双引号时字符串中可以包括单引号  
* 转义字符,使用`\` 反斜杠来输入特殊字符,  `\t` 制表位, `\n` 换行符, `\\` 反斜杠
* 原始字符串,在字符串前面加一个`r`,例如 `r'abc'` ,可以完全忽略字符串中的所有反斜杠
* 多行字符串,对于有换行的字符串可以更为方便的输入
* raw字符串,在字符串前加一个 'r' 代表rawString
   1. 取消对反斜杠\ 的转义
   2. 如果一个字符串包含很多需要转义的字符，对每一个字符都进行转义会很麻烦。为了避免这种情况,我们可以在字符串前面加个前缀r，表示这是一个 raw 字符串，里面的字符就不需要转义了。
* 多行字符串`'''abc'''`,字符串可以跨过多行
   1. 三引号还会对其内的单双引号起作用
   2. 可以将raw字符串和多行字符串结合起来使用
* 字符串的编码类型
   1. python中字符串默认采用的ASCII编码，
   2. 如果要显示声明为unicode类型的话，需要在字符串前面加上'u'或者'U' `print(u'字符串')`
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
字符串可以使用切片以及 `in `  `not in` 操作符,用来比较前一个字符串是否在后一个字符串中间
  
## 7.2. 常用的字符串方法

### 7.2.1. 大小写及内容检测方法
   * `upper()` 和 `lower()` 返回一个**新字符串**,将原本字符串中所有字母转变为大写/小写
   * `name.title()`  标题方法, 将字符串的单词第一个字母大写  
   * `isupper()` 和 `islower()` 返回布尔值,如果这个字符串不为空且全部字符为大写/小写则返回`True`
   * 其他的 isX 方法,返回True的情况
     * isalpha()  非空且只包含字母
     * isalnum()  非空且只包含字母和数字
     * isdecimal() 非空且只包含数字字符
     * isspace()  非空且只包含空格,制表符,换行
     * istitle()  非空且只包含以大写字母开头,后面都是小写字母的 `单词` 及可以包含空格及数字
### 7.2.2. 开头结尾检测方法  
   startswith() 和 endswith()
   以传入的字符串为开始/结尾,则返回True

### 7.2.3. 组合与切割方法  
   join() 和 split()    
    join()是将一个字符串列表连接起来的方法,并在字符串中间插入调用`join`的字符串  
    `','.join(['a','b','c'])`   返回 ` 'a,b,c' `  
    split()则相反,在被拆分的字符串上调用,默认以各种空白字符分隔  
    `'My name is'.split()`   返回 `['My','name','is']`  
    常被用来分割多行字符串  ` spam.split('\n')`

### 7.2.4. 对齐方法  
   rjust()  ljust()  center()
   在一个字符串上调用,传入希望得到的字符串的长度,将返回一个以空格填充的字符串  
   分别代表左对齐,右对齐  
   `'a'.rjust(5)`  返回  `'    a'`   
   `'a'.ljust(5)`  返回  `'a    '`
   可以输入第二个参数改变填充字符  
   `'a'.ljust(5,'*')`  返回  `'a****'`

### 7.2.5. 清除对齐方法  
   `strip()  rstrip() lstrip()  `  
   在左右删除空白字符  
   传入参数指定需要删除的字符  注:这里第二个参数无视字符出现顺序  
   `'abcccba'.strip('ab')` 与
   `'abcccba'.strip('ba')` 作用相同

### 7.2.6. 识别转换方法 **str.extract()**
和pandas组合使用,拆解字符串
```py

# Extracting ID into ['faculty','major','num']
id_ext = df['id'].str.extract('(\w{3})-(\w{2})(\d{2})',expand=True)
id_ext.columns = ['faculty','major','num']


# Extracting 'stats' into ['sex', 'age']
stats_ext = df['stats'].str.extract('(\w{1})_(\d{2})years',expand=True)
stats_ext.columns = ['sex', 'age']
```
   
### 7.2.7. str.get_dummies()

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


# 8. Python 的类

python 的类通过`class`定义 , python的类名一般以大写字母开头

类的构造函数  
`def __init__(self,<other parameter>):`  
`__init__` 是保留的构造函数名, `self` 是保留的参数, 类的所有方法都需要有`self`参数  

# 9. Python 的文件操作 

## 9.1. 打开文件

`with`关键字  
`open('路径')`  
`as [别名]  `   

读取一个文件的字符串  
```python
#这个文件对象只在with的Block里面有效
with open('pi_digits.txt') as file_object:
  contents = file_object.read()         
  # read() 读取整个文件
  
  #按行来读取文件
  for line in file_object:
    print(line)
  #将文件读取到一个list里
  lines = file_object.readlines()
for line in lines:  #可以在block外读取文件内容
  print(line.rstrip()) # 可以用rstrip方法来清除文件的换行符

```

## 9.2. 写入文件

要想写入文件,需要在文件对象创建的时候指定`'w'`参数,或者`'a'`参数    
```python
with open(filename, 'w') as file_object: 
  # 使用 write()方法来写入内容
  file_object.write("I like programming.\n") # 记得自己输入换行符

# 使用'a'参数来代表追加模式, 文字将输入到文件末尾  
with open(filename, 'a') as file_object: 

```

## 9.3. 结构化读取文件
使用 enumerate 可以按行获取文件内容  
`for j, data in enumerate(openfile) `

```py
with open('animal.txt', 'r') as openfile:
  for j, data in enumerate(openfile):
    if j % n == 0:
      print(f'Line number {str(j)}. Content: {data}')
```

# 10. 正则表达式 re包 

要在python中使用正则表达式, 需要导入`re`包  
`import  re`  

官方文档[https://docs.python.org/3/library/re.html]

## 10.1. 使用正则表达式的基础函数

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

## 10.2. 正则表达式-单字符

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

## 10.3. 正则表达式-多次匹配

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

# 11. Python 的包 环境管理 

[官方推荐文档]<https://packaging.python.org/guides/tool-recommendations/>  

## 11.1. Python的包管理
包管理工具有很多

### 11.1.1. distutils 和 setuptools

distutils是 python 标准库的一部分  
用于方便的打包和安装模块  是常用的 setup.py 的实现模块  
setuptools 是对 distutils 的增强，尤其是引入了包依赖管理  


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



### 11.1.2. pip 

pip是目前最流行的Python包管理工具，它被当作easy_install的替代品，但是仍有大量的功能建立在setuptools之上。  

* Python 2.7.9及后续版本：默认安装，命令为pip
* Python 3.4及后续版本：默认安装，命令为pip3


pip的使用非常简单，并支持从任意能够通过 VCS 或浏览器访问到的地址安装 Python 包  

* 安装:  pip install SomePackage 
* 卸载:  pip uninstall SomePackage 

* pip list 查看已安装包的列表
* pip freeze 另一种查看方法
  * `pip freeze > requirements.txt` 将输出存入文件 可以使别人安装相同版本的相同包变得容易
  * `pip install -r requirements.txt`


## 11.2. 环境管理

在开发Python应用程序的时候，系统安装的Python3只有一个版本：3.4。所有第三方的包都会被pip安装到Python3的site-packages目录下。

如果我们要同时开发多个应用程序，那这些应用程序都会共用一个Python，就是安装在系统的Python 3  
如果应用A需要jinja 2.7，而应用B需要jinja 2.6怎么办  

有多种虚拟环境配置方法  

### 11.2.1. venv  

Source code: Lib/venv/  
一般已经安装在了较新的 python 版本中了  因为是从 3.3 版本开始自带的，这个工具也仅仅支持 python 3.3 和以后版本  
创建一个轻量级虚拟环境, 与系统的运行环境相独立, 有自己的 Python Binary  

使用 `venv` 命令进行虚拟环境操作  

```shell
# vene ENV_DIR
python3 -m venv tutorial-env
python3 -m venv /path/to/new/virtual/environment


# 激活虚拟环境
source tutorial-env/bin/activate

```

### 11.2.2. virtualenv

virtualenv 是目前最流行的 python 虚拟环境配置工具
* 同时支持 python2 和 python3
* 可以为每个虚拟环境指定 python 解释器 并选择不继承基础版本的包。

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

### 11.2.3. virtualenvwrapper

`pip install virtualenv virtualenvwrapper`  

virtualenvwrapper 是对 virtualenv 的一个封装，目的是使后者更好用
使用 shell 脚本开发, 不支持 Windows  

它使得和虚拟环境工作变得愉快许多
* 将您的所有虚拟环境在一个地方。
* 包装用于管理虚拟环境（创建，删除，复制）。
* 使用一个命令来环境之间进行切换。


```shell


#设置环境变量 这样所有的虚拟环境都默认保存到这个目录
export WORKON_HOME=~/Envs  
#创建虚拟环境管理目录
mkdir -p $WORKON_HOME


# 每次要想使用virtualenvwrapper 工具时，都必须先激活virtualenvwrapper.sh
find / -name virtualenvwrapper.sh #找到virtualenvwrapper.sh的路径
source 路径 #激活virtualenvwrapper.sh

# 创建虚拟环境  
# 该工具是统一在当前用户的envs文件夹下创建，并且会自动进入到该虚拟环境下  
mkvirtualenv ENV
mkvirtualenv ENV  --python=python2.7

# 进入虚拟环境目录  
cdvirtualenv

Create an environment with `mkvirtualenv`

Activate an environment (or switch to a different one) with `workon`

Deactivate an environment with` deactivate`

Remove an environment with`rmvirtualenv`

# 在当前文件夹创建独立运行环境-命名
# 得到独立第三方包的环境，并且指定解释器是python3
$ mkvirtualenv cv -p python3

# 进入虚拟环境  
source venv/bin/activate  

#接下来就可以在该虚拟环境下pip安装包或者做各种事了，比如要安装django2.0版本就可以：
pip install django==2.0

```

**其他命令**:
* workon `ENV`          : 启用虚拟环境
* deactivate            : 停止虚拟环境
* rmvirtualenv `ENV`    : 删除一个虚拟环境
* lsvirtualenv          : 列举所有环境
* cdvirtualenv          : 导航到当前激活的虚拟环境的目录中，比如说这样您就能够浏览它的 site-packages
* cdsitepackages        : 和上面的类似，但是是直接进入到 site-packages 目录中
* lssitepackages        : 显示 site-packages 目录中的内容

