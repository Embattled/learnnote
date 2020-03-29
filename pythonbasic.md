# 1. Python基础

## 1. Python的语法

* python每行代码不需要分号
* 变量用小写字母开头
* `'单引号中的内容代表字符串'`
* `#` 号代表注释
* 程序运行中使用<kbd>Ctrl</kbd>+<kbd>C</kbd> 可以立即终止程序

## 2. 常用的最基础的函数
* `myname=input()` 接受键盘输入的一个字符串,结果存储到变量`myname`
* `len()`  括号中传入一个字符串
* 类型转换函数
  * `str()`  括号中传入整型数字,返回字符串
  * `int()`  `float()` 分别为传入数字的字符串,并将类型转换为数字

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
python没有大括号,以缩进代表代码段  
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