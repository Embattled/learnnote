# 1. Data Types

除了一些 build-in data types, in particular, dict, list, set and frozenset, and tuple.   
在 python STL中还有很多特殊的数据类型  


最典型的例如时间序列类型  
DataTypes中的数据类型:
* collections : 提供了几个特殊的容器


# 2. copy

专门用来拷贝的小库, 在python中赋值语句只会进行引用传递, 这个库中提供了两个对应的深浅拷贝函数

import copy

* copy.copy(x)  返回一个 shallow copy of x
  * 只会拷贝最浅1层的数据
  * 深层数据, 如 `[1,2,[a,b] ] ` 中的list`[a,b]`被改变的时候, 依然会反映到所有拷贝中
* copy.deepcopy(x)  返回深拷贝
  * 递归拷贝到最深层
  * 如果某个对象是递归对象, 即包含了自己的引用, 会报错
  * 如果有对象间共享数据的话就不要用这个




# 3. datetime — Basic date and time types


# 4. collections

* 特殊的容器, 作为python内建类型的特殊替代场景

## 4.1. Counter 

A counter tool is provided to support convenient and rapid `tallies`, dict subclass for counting hashable objects.

* `class collections.Counter([iterable-or-mapping])`  用一个可以迭代的对象建立一个 Counter
* 对象的元素必须是 hashable
* Counter 本身是 dict 的子类

```py

# Counter 本身的初始化有两种, 可迭代的值初始化, 或者直接用类似于字典的初始化
c = Counter()                           # a new, empty counter
c = Counter('gallahad')                 # a new counter from an iterable
c = Counter({'red': 4, 'blue': 2})      # a new counter from a mapping
c = Counter(cats=4, dogs=8)             # a new counter from keyword args

# Counter 属于一个 dict 的子类, 很重要的改动就是可以直接访问不存在的 key
c = Counter(['eggs', 'ham'])
c['bacon']                              # count of a missing element is zero

```


Counter 方法:
* elements()  : 返回被添加的所有元素的迭代器, 每一个元素都会出现 count 次, 而 count 小于 1 的会被无视, 元素的顺序是按照它们第一次被添加的顺序
* `most_common([n])` : 返回最常出现的元素以及对应的出现次数
  * 返回的内容是键值对的 list, e.g. `[('a', 5), ('b', 2), ('r', 2)]`
  * 出现次数相同的元素按照第一次添加的顺序
  * 

## 4.2. collections.abc

Abstract Base Classes for Containers:
* 正式的来说, 这个包属于 collections 的一部分, 但是内容区别较大
* 提供了很多 抽象基类, 用于对各种类型进行功能检测


# enum — Support for enumerations

在 C 语言中最基础的类型在 python 中被实装到了库中  

An enumeration:
* is a set of symbolic names (members) bound to unique values
* can be iterated over to return its canonical (i.e. non-alias) members in definition order
* uses call syntax to return members by value
* uses index syntax to return members by name

在 python 中, 通过 `from enum import Enum` 导入后, 有两种方法定义一个枚举

```py
# class syntax
class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

# functional syntax
Color = Enum('Color', ['RED', 'GREEN', 'BLUE'])
```

同 C 语言不同, python 的枚举类的每一个对象, 都有 name 和 value 两个属性, name 就是定义时候的枚举成员名称, 而值就是索引值, 需要特别注意索引是 1-index   
```py
list(Color)
[<Color.RED: 1>, <Color.GREEN: 2>, <Color.BLUE: 3>]
```



