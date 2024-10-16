# 1. Runtime Services

该分类的模组用于提供 Python 解释器和其运行环境, 也就是python 解释器和操作系统之间的交互  

# 2. sys python解释器服务

与 python 解释器密切相关的一些函数与变量  

可以用于配置 python 整个程序的运行环境  
* 经常与 os 包搞混
* 该包的内容更贴近于 python system

## 2.1. 标准流对象

* sys.stdin     stdin 用于所有交互式输入 （包括`input()`）
* sys.stdout    stdout用于所有输出( 包括`print()`, `Input()`的提醒信息, 以及)
* sys.stderr    error messages 输出
这些标准流对象都和 `open()` 返回的对象类似, 都可以进行作为参数传入相关函数  

可以将输出流或者错误流赋值 `None` 或者 linux 标准垃圾箱 `/dev/null` 来减少输出  

以下对象保存了程序在一开始运行(也就是系统赋予的)的 std* 的初始值  
可以确保在 std* 被重定向后仍然能重定向回来  
* `sys.__stdin__`
* `sys.__stdout__`
* `sys.__stderr__`

## 2.2. sys.path 库查找路径

`sys.path` 是一个 string 的 list 对象  
* 会参照系统的 `PYTHONPATH` 的内容来初始化
* 执行 python 解释器的时候会自动加入各种安装好的包的路径以及 STL 路径


`sys.path[0]` 是特殊的, 用于存储调用 python 解释器的脚本的路径  
* 如果是以交互式方式启动的 python 解释器或者从标准输入流得到的 python 代码, 则该项目为空字符串
* 几个默认路径, 通过 `site` 包被自动导入  
  * `/lib/python310.zip`
  * `/lib/python3.10`                   python STL
  * `/lib/python3.10/lib-dynload`
  * `/lib/python3.10/site-packages`
* 通过往该路径加入新的元素可以动态扩充包的搜索路径  

## 2.3. sys.platform 平台识别



# 3. sysconfig python解释器配置

用于指定 python 自己的相关控制参数  

# 4. warnings — Warning control

程序向使用者发出警告的模块

通常使用 `warn()` 函数来触发, 并且将警告信息发送到 sys.stderr 中

警告的处理可能会根据警告的类别, 文本消息, 以及发出警告的源位置而有所不同, 但通常对于同一个位置产生的警告会抑制重复提醒  
警告本身会经过两个阶段: 判定和输出
* 判定是由一个单独的模块 `filterwarnings()` 来进行的, 通过向其中添加规则来使得程序进行过滤  
* 警告的打印也是由两个 函数 showwarning  formatwarning 
  * 除此之外 logging 模组也支持将警告类记录到日志里


## 4.1. Warning Categories

内置的一些 实现为异常的 警告类, 尽管属于 `built-in exceptions`, 但他们的机制属于 warning

同异常一样, 警告也可以自定义实现, 同时所有的 Warning 都必须属于 `Warning` 主类的子类


类别说明
* Warning                 : 所有警告的父类, 同时 Warning 本身是 Exception 类的子类
* UserWarning             : 凡是由 warn() 调用的警报都 默认属于 该类
...

## 4.2. The Warnings Filter - 警报过滤器

Filter 控制一个产生的警备是否被 忽略/显示/转为异常(被raise).  

从概念上, warnings filter 会维护一个 ordered list 用于描述 filter 的规范
* 一个传入的 warning 会按照 list 的顺序依次匹配, 直到找到匹配项确定 warning 的行为
* list 的每一个元素都是一个 tuple of the form (action, message, category, module, lineno)

filter 的每个元素的构成 (action, message, category, module, lineno)
* `action` : one of the following strings 
  * default   : 按照类似于 exception 一样打印 the first occurrence of matching warnings for each location (module + line number) 
  * error     : 将 matching warnings 转为 exceptions
  * ignore    : matching warnings 将会被忽略
  * always    : 总是打印该 warning 
  * module    : print the first occurrence of matching warnings for each module where the warning is issued (regardless of line number)
  * once      : 打印 警报发生的第一个 occurrence, 无视 location
* `message`     : a string containing a regular expression that the start of the `warning message` must match. 大小不敏感, filter message 是 warning message 在开头必须包含的文字字符串.
* `category`    : class, 用于 Filter Match 的主要参数
* `module`      : a string containing a regular expression that the start of the fully qualified module name must match. 大小写敏感, 产生 warning 的 module 必须匹配该 module 表达式
* `lineno` is an integer that the line number where the warning occurred must match, or 0 to match all line numbers.
通过以上 4 个匹配的项目, 以及一个 action 来决定 warning 最终的表现, 如果warning 没有被匹配, 则使用 default action


### 4.2.1. Describing Warning Filters

### 4.2.2. Default Warning Filter

默认行为下 Python 对于 Warning 的表现  

release 版本的 默认 filter 为以下, 而 debug 版本的 filter 默认为空
```py
default::DeprecationWarning:__main__
ignore::DeprecationWarning
ignore::PendingDeprecationWarning
ignore::ImportWarning
ignore::ResourceWarning
```
### 4.2.3. Overriding the default filter - 覆盖默认 filter



## 4.3. Available Functions - 完整包函数

快速上手函数:
* `warnings.simplefilter(action, category=Warning, lineno=0, append=False)`
  * 快速添加一个简单的 filter 到 warning filter list
  * 参数的含义与 `filterwarnings()` 相同, 但是不再需要正则表达式的 `message` 和 `module` 参数
  * 即, 任意模组的任意消息的警告, 只要满足 category 和 lineno 相同, 就会被匹配
* `warnings.resetwarnings()`
  * 还原所有 warnings filter 为 default
  * 会取消掉所有 `filterwarnings() 和 simplefilter()` 的调用, 包括 `-W` 的命令行选项

# 5. dataclasses — Data Classes

该模块提供了一个 装饰器 decorator 和几个函数, 用于自动将生成的 special methods 添加到 用户定义的类中
* 所谓的 special methods 举例有 : `__init__()` 和 `__repr__()`

描述于 PEP 557 Python 3.7 


快速示例, 通过 PEP 526 定义的类型注释 (annotations) 和 @dataclass 来隐式的定义构造函数

```py
from dataclasses import dataclass

@dataclass
class InventoryItem:
    """Class for keeping track of an item in inventory."""
    name: str
    unit_price: float
    quantity_on_hand: int = 0

    def total_cost(self) -> float:
        return self.unit_price * self.quantity_on_hand

    # 这个函数是被隐式的定义的
    def __init__(self, name: str, unit_price: float, quantity_on_hand: int = 0):
        self.name = name
        self.unit_price = unit_price
        self.quantity_on_hand = quantity_on_hand
```


## 5.1. Module contents


### 5.1.1. Field 类 概念



`class dataclasses.Field`
* 一个 Field 对象 定义了一个 field. 这些 对象都是隐式的定义的, 并作为 `dataclass.fields()` 接口的返回值
* 用户永远不应该 自己定义一个 Field 实例
* field 保有的属性有
  * name
  * type
  * `default, default_factory, init, repr, hash, compare, metadata, kw_only` have the identical meaning and values as they do in the `field()` function.
* 除此之外还有一些用于正确工作的私有属性

`dataclasses.fields(class_or_instance)`
* 将参数转化成  `tuple of Field objects`, 接受的参数类型为  dataclass or instance of a dataclass
* 如果参数不是 dataclass or instance of a dataclass 会报错


`dataclasses.field(*, default=MISSING, default_factory=MISSING, init=True, repr=True, hash=None, compare=True, metadata=None, kw_only=MISSING)`
用于辅佐 dataclass 的定义, 通过替换 annotation 字段为 `field()` 函数可以实现对 field 追加额外信息
```py
@dataclass
class C:
    x: int
    y: int = field(repr=False)
    z: int = field(repr=False, default=10)
    t: int = 20
# The class attribute C.z will be 10, the class attribute C.t will be 20, and the class attributes C.x and C.y will not be set.
```
* 参数:
  * init: If true (the default), this field is included as a parameter to the generated `__init__()` method.
  * repr: If true (the default), this field is included in the string returned by the generated `__repr__()` method.


### 5.1.2. 装饰器 @dataclasses.dataclass

`@dataclasses.dataclass(*, init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False, weakref_slot=False)`
* 一个装饰器 用于自动生成 类的 特殊方法
* 定义了该装饰器的类会自动查找类的 `field`s
  * field : 是一个 拥有 type annotation 的类的变量
  * 尽管 field 包含 type annotations, 但并不会实际上检查 变量的类型, 除了两个例外
* `field`s 在生成的方法中的顺序 同 他们在类定义中的顺序一致
* 该装饰器会自动给类添加许多 `dunder` 方法, 而如果这些方法已经存在于类中, 则最终的行为取决于 装饰器参数
  * 装饰器的默认参数调用有三种写法


```py
@dataclass
@dataclass()
# 完整的默认参数调用
@dataclasses.dataclass(*, init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False, weakref_slot=False)
```

### 5.1.3. 类型转换函数


`dataclasses.asdict(obj, *, dict_factory=dict)`
* 把 dataclass obj 转换为字典, 通过使用 工厂函数 `dict`
* dataclass 会转换成其 fileds 的字典
  * dataclasses, dicts, lists, tuples 会递归的嵌入
  * 其他的类型会直接进行 深拷贝 
* obj 必须是 dataclass 的实例, 否则会报错

```py
@dataclass
class Point:
     x: int
     y: int

@dataclass
class C:
     mylist: list[Point]

# 查验 asdict 的结果
p = Point(10, 20)
assert asdict(p) == {'x': 10, 'y': 20}

c = C([Point(0, 0), Point(10, 4)])
assert asdict(c) == {'mylist': [{'x': 0, 'y': 0}, {'x': 10, 'y': 4}]}

# 若要避免 asdict 的深拷贝, 可以使用特殊的构造方法
{field.name: getattr(obj, field.name) for field in fields(obj)}
```


`dataclasses.astuple(obj, *, tuple_factory=tuple)`
* 把 dataclass obj 转换为 tuple, 通过使用 工厂函数 `tuple`
* dataclasses, dicts, lists, and tuples are recursed into.
* Other objects are copied with copy.deepcopy().
* 同样的 obj 只接受 dataclass


```py
# 同 asdict 相比就是失去了 变量名作为 字典的  key 的能力
assert astuple(p) == (10, 20)
assert astuple(c) == ([(0, 0), (10, 4)],)

# 浅拷贝
tuple(getattr(obj, field.name) for field in dataclasses.fields(obj))
```




# 6. abc - Abstract Base Class

该包提供了 Python 用于定义虚基类 ABC 的基础设置, 在 PEP3119 中被引述.  
此外 numbers 模组中的层次数字类是基于 ABC 的, 可以参阅 PEP3141

collections 中的一部分类也是基于 ABC的, 同时 可以通过访问 collections.abc 来访问具体的虚基类  

abc 的模块提供了所谓的 metaclass `ABCMEta` 用于具体的定义一个 ABC, 以及一个辅助类 `ABC` 用于通过继承来定义 ABC


虚基类的多重继承可能会导致冲突, 因此要小心
```py

# New in version 3.4
class abc.ABC
# 用于通过继承来定义虚基类, 具体使用方法为

from abc import ABC
class MyABC(ABC):
  pass

#
```
# 7. traceback 用于异常溯源

# 8. inspect — Inspect live objects

审查模组
* 获取活动对象的信息, (模组, 类, 方法, 函数等)
    1. examine the contents of a class
    2. retrieve the source code of a method
    3. extract and format the argument list for a function
    4. get all the information you need to display a detailed traceback.


## 8.1. 类型以及成员



### 8.1.1. 对象自带属性
在该模组的文档中有 python 所有对象的自带属性的[说明](https://docs.python.org/3/library/inspect.html)

## 8.2. 获取源代码

* getsource(obj)

## 8.3. 




# 9. site Site-specific configuration hook¶

这是一个会被 python 解释器自动导入的包  
* This module is automatically imported during initialization.
* The automatic import can be suppressed using the interpreter’s -S option.
* 用于自动导入几个 python 解释器默认的路径  
