
# 1. Compound Statements - 复合语句

复合语句 (包含了其他语法的代码组), 通常复合语句都会有多行  

## 1.1. The if statement

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

## 1.2. The while statement

内容同样不需要括号

```py
while 条件表达式：
    代码块

else:
    pass
```

## 1.3. The for statement

## 1.4. The try statement


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


- 逻辑流程控制
  - `break`和`continue` 同C语言是一样的
  - Python中, while 和for 后也可以紧跟着一个 else 代码块
    - 当循环条件为 False 跳出循环时, 程序会最先执行 else 代码块中的代码
    - 但是使用 `break` 跳出当前循环体之后, 该循环后的 `else` 代码块**也不会被执行**



## 1.5. The with statement - with 表达式

通过 context manager 的方式来包装块的执行

```py
with_stmt          ::=  "with" ( "(" with_stmt_contents ","? ")" | with_stmt_contents ) ":" suite
with_stmt_contents ::=  with_item ("," with_item)*
with_item          ::=  expression ["as" target]
```

自底向上说明该语法:
* `with_item` : context expression 会被执行, 其结果会作为 context manager, `["as" target]` 用于给 context manager 赋予名称


## 1.6. The match statement - python3.10 加入

在 python 3.10 以前的版本使用该语句会报错, 向下兼容性很差

```sh
match_stmt   ::= 'match' subject_expr ":" NEWLINE INDENT case_block+ DEDENT
subject_expr ::= star_named_expression "," star_named_expressions?
                 | named_expression
case_block   ::= 'case' patterns [guard] ":" block
```

## 1.7. Function definitions - 函数的定义

## 1.8. Class definitions - 类的定义
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




## 1.9. Coroutines - 协程函数的定义


## 1.10. Type parameter lists 

Python 3.12 新功能



## 1.11. self

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

## 1.12. 类的变量

1. 类变量: 类变量指的是在类中, 但在各个类方法外定义的变量。
   - 所有类的实例化对象都同时共享类变量, 即在所有实例化对象中是作为公用资源存在的
   - 既可以使用类名直接调用, 也可以使用类的实例化对象调用
2. 实例变量: 在任意类方法内部, 以`self.变量名`的方式定义的变量
   - 只作用于调用方法的对象。
   - 只能通过对象名访问, 无法通过类名访问。
3. 局部变量 : 类方法中普通方法定义, 不使用 `self.` 来定义的变量
   - 函数执行完成后, 局部变量也会被销毁。



## 1.13. 类的描述符

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

## 1.14. 类的封装

- Python 并没有提供 public、private 这些修饰符
    默认情况下, Python 类中的变量和方法都是公有（public）的, 它们的名称前都没有下划线（_）；
- 如果类中的变量和函数, 其名称以`双下划线`“__”开头, 则该变量（函数）为私有变量（私有函数）, 其属性等同于 private。
  - python 的私有变量的封装方法不是结构性的, 只是在内部将私有变量的名称更改了
  - 对于一个变量 `__私有变量` 在执行过程中实际的变量名变成了 `_类名__私有变量` 因此仍然可以在外部访问
- 用 `类对象.属性` 的方法访问类中的属性是不妥的, 破坏了类的封装性

### 1.14.1. property()

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

### 1.14.2. @property 装饰器

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

## 1.15. 类的继承和多态

- 类的继承, 在定义子类的时候, 将父类放在子类之后的圆括号即可
  - `class 类名(父类1, 父类2, ...)：`
  - 所有类都隐式继承自 `object` 类
  - python 支持多继承, 大部分的对象语言都不允许多继承
    - 对于多个父类中的同名方法, 以最早出现的父类为准

### 1.15.1. super
  
- 子类如果定义了自己的构造方法, 则里面必须要调用父类的构造方法
- 在子类中的构造方法中, 调用父类构造方法的方式有 2 种, 分别是：
  - 类可以看做一个独立空间, 在类的外部调用其中的实例方法, 可以向调用普通函数那样, 只不过需要额外备注类名（此方式又称为未绑定方法）；
  - 使用 super() 函数。但如果涉及多继承, 该函数只能调用`第一个直接父类的构造方法`。
  - 及到多继承时, 在子类构造函数中, 调用第一个父类构造方法的方式有以上 2 种, 而调用其它父类构造方法的方式只能使用未绑定方法。

`super().__init__(self,...)`

### 1.15.2. MRO Method Resolution Order