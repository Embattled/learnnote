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


## 4.2. collections.abc

Abstract Base Classes for Containers:
* 正式的来说, 这个包属于 collections 的一部分, 但是内容区别较大
* 提供了很多 抽象基类, 用于对各种类型进行功能检测


