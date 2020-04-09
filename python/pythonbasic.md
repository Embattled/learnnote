# 1. Python基础

## 1. Python的语法

* python每行代码不需要分号
* 变量用小写字母开头
* `'单引号中的内容代表字符串'`
* `#` 号代表注释
* `"""` 三个双引号中间的内容代表多行注释  
* 程序运行中使用<kbd>Ctrl</kbd>+<kbd>C</kbd> 可以立即终止程序
* python代码第一行的声明,缺少这个声明仍然可以在idle中运行脚本,但是不能从系统命令行中运行
  * windows 下 `#! python3`
  * OS X 下, `#! /usr/bin/env python3`
  * Linux 下, `#! /usr/bin/python` 
## 2. 常用的最基础的函数
* `myname=input()` 接受键盘输入的一个字符串,结果存储到变量`myname`
* `len()`  括号中传入一个字符串
* 类型转换函数
  * `str()`  括号中传入整型数字,返回字符串
  * `int()`  `float()` 分别为传入数字的字符串,并将类型转换为数字
* 使用命令行参数  命令行参数存储在 `sys.argv` 中,以列表的形式,第一个项是文件名,从第二项开始是第一个命令行参数

## 3. Python的操作符与变量
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

## 4. 代码段与条件表达式
python没有大括号,以缩进代表代码段,告诉python该语句属于哪一个代码块    
缩进规则在大多数情况下有用,可以使用 `/` 来使得一条指令可以跨越到下一行,同时使下一行的缩进无效,增加代码可读性  
使用`:` 来代表内容的开始

### `if` 表达式
```python
if 表达式:
  pass #内容
elif 表达式:
  pass
else:
  pass
```
### `while` 表达式
内容同样不需要括号

`break`和`continue` 同C语言是一样的

### `for` 表达式

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

## 5. python的包

使用 `import <name>` 来导入一个包,可以使用其中的函数  
在使用的时候要加上包名 `name.function()`  
使用 `from <name> import *` 可以使得调用包中函数不再需要包名  

### 常用的包
* 包的下载 , 使用pip来下载及管理包
  * `pip install ModuleName` 下载一个包  
  * `pip install -u ModuleName` 下载一个已经安装包的更新版本  
* random
  * `random.randint(*,*)` 给出两个参数整数之间的一个随机整数
* sys
  * `sys.exit()` 提前结束程序


## 6. python的函数及作用域

### 1. 定义
```python
`def functionName(): #python的函数参数不需要类型`
  functionpass
  return None
  # python的所有函数都需要返回值,就算不手动写出也会在幕后给没有 return 的函数添加 return None
```
### 2. 参数
* 位置识别,同C语言一样
* **关键字参数**,根据调用时加在参数前面的关键字来识别,通常用于可选参数
  * 例如`print()` 函数的`end`和`sep`用来指定参数末尾打印什么,参数之间打印什么
  * `print('Hellow',end='')` 在打印后不会自动换行
  * `print('a','b','c')` 在参数之间会默认隔一个空格
  * `print('a','b','c',seq=',')` 会输出 **a,b,c** 
  
### 3. 作用域

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

## 7. 函数的异常处理

如果 `try` 子句中的代码发生了错误，则程序立即到 `except` 中的代码去执行  

因此将`try` 放到函数中和直接放到代码段中会有不同的效果，会影响程序执行的流程，因此一般**将异常封装在函数里**.

```python
try:
   return 42/eggs
except ZeroDivisionError:
   print('divide zero')
```

# 2. python 的列表和元组

## 1. 列表类型

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
  
## 2. 列表的使用
python的列表总是动态的:
```python
catName=[]
while True:
    catName=catName+[name]
for name in catName:
    print(name)
```

### 列表的循环
` for i in range(len(someList))`

### 列表的查找
`in `和 `not in` 操作符  
可以确定一个值是否在列表中
```python
'howdy' in ['hello','hi','howdy']
返回值为 True 或者 False
```

### 列表的赋值
使用列表可以同时为多个变量赋值  
`size,color,dispo=cat`  
变量的数目和列表的长度必须相等,否则报错  
`cat=['fat','black','loud']`

### 增强的赋值操作
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

### 列表类型的方法

* `index()` : `spam.index('hello')` 传入一个值,若该值存在于列表中,返回**第一次出现的下标**
* `append()` :`spam.append('hello') ` 将值添加到列表末尾
* `insert()`: `spam.insert(1,'chicken')` 将值添加到参数下标处
* `remove()`: `spam.remove('bat') ` 传入一个值，将该值从列表中删除，若该值不在列表中，返回`ValueError`错误,只会删除第一次出现的位置
  * 若知道删除的下标,就用 `del spam[2]`
  * 若知道要删除的值,就用`spam.remove('a')`
* `sort()`: 
  * `spam.sort()`数值的列表或者字符串的列表可以使用该方法
  * `spam.sort(reverse=True)` 指定反向排序
  * `sort()` 是当场排序,不是返回值,不能对混合值列表进行排序
  * `sort()` 是按照ASCII表排序,因此在大写是排在小写字母之前
    * 可以使用`spam.sort(key=str.lower)`使方法把所有的表项当作小写

## 3. 字符串和元组
### 1. 字符串类型
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

### 元组数据类型 
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

## 4. 关于引用
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

# 3. 字典和结构化数据
## 1. 字典
同列表一样,字典是许多值的集合,但不同于列表的下标,字典的索引使用许多不同的数据类型,不只是整数

字典的定义使用花括号 `{}`
```python
myCat={ 'size':'fat','color':'gray','disposition':'loud' }
```
输入 `myCat['size']` 会输出`'fat'`
当然也可以使用数字作为索引,但是完全不受限制,不必从0开始

字典的内容完全不做排序,没有**第一个**的概念

## 2. 字典的使用
字典有三个打印方法
* `keys()`  :返回键值
* `values()`:具体值
* `items()` :键-值对
这些方法返回的不是真正的列表，不能被修改，不能使用 append() 方法，但可以用于 `for` 循环

可以使用 `list(spam.keys())` 将返回的值转换为列表

### get() 方法
使用`spam.get('索引',备用值)`可以安全的通过索引访问一个值,当这个索引在字典中不存在时返回备用值  
若不使用`get`方法则会报错

### setdefault() 方法
`setdefault('索引',值)` 可以安全的初始化一个键,当该键存在则返回键值,键不存在则设定为第二个参数

### 字典的优化打印
自带的`print()`函数会把字典打印在一行  
使用`pprint`模块,可以优化打印
```python
someDictionay={}
pprint.pprint(someDictionary) # 可以每个键值对打印在一行
string=pprint.pformat(someDictionary) #返回一个和 上一行打印内容相同的字符串
print(string) # 效果和第一行相同
```

# 4. 字符串操作

## 1. 字符串的输入
1. python的字符串使用 `''` 单引号输入  
2. 也可以使用 `"" `双引号输入,区别在于使用双引号时字符串中可以包括单引号  
3. 转义字符,使用`\` 反斜杠来输入特殊字符,  `\t` 制表位, `\n` 换行符, `\\` 反斜杠
4. 原始字符串,在字符串前面加一个`r`,例如 `r'abc'` ,可以完全忽略字符串中的所有反斜杠
5. 多行字符串,对于有换行的字符串可以更为方便的输入
```python
'''
this is a long string with multiline.
'''
```
6. 字符串可以使用切片以及 `in `  `not in` 操作符,用来比较前一个字符串是否在后一个字符串中间
  
## 2. 常用的字符串方法

1. 大小写及内容检测方法
   * `upper()` 和 `lower()` 返回一个**新字符串**,将原本字符串中所有字母转变为大写/小写
   * `isupper()` 和 `islower()` 返回布尔值,如果这个字符串不为空且全部字符为大写/小写则返回`True`
   * 其他的 isX 方法,返回True的情况
     * isalpha()  非空且只包含字母
     * isalnum()  非空且只包含字母和数字
     * isdecimal() 非空且只包含数字字符
     * isspace()  非空且只包含空格,制表符,换行
     * istitle()  非空且只包含以大写字母开头,后面都是小写字母的 `单词` 及可以包含空格及数字
2. 开头结尾检测方法  
   startswith() 和 endswith()
   以传入的字符串为开始/结尾,则返回True

3. 组合与切割方法  
   join() 和 split()    
    join()是将一个字符串列表连接起来的方法,并在字符串中间插入调用`join`的字符串  
    `','.join(['a','b','c'])`   返回 ` 'a,b,c' `  
    split()则相反,在被拆分的字符串上调用,默认以各种空白字符分隔  
    `'My name is'.split()`   返回 `['My','name','is']`  
    常被用来分割多行字符串  ` spam.split('\n')`

4. 对齐方法  
   rjust()  ljust()  center()
   在一个字符串上调用,传入希望得到的字符串的长度,将返回一个以空格填充的字符串  
   分别代表左对齐,右对齐  
   `'a'.rjust(5)`  返回  `'    a'`   
   `'a'.ljust(5)`  返回  `'a    '`
   可以输入第二个参数改变填充字符  
   `'a'.ljust(5,'*')`  返回  `'a****'`

5. 清除对齐方法  
   strip()  rstrip() lstrip()  
   在左右删除空白字符  
   传入参数指定需要删除的字符  注:这里第二个参数无视字符出现顺序  
   `'abcccba'.strip('ab')` 与
   `'abcccba'.strip('ab')` 作用相同