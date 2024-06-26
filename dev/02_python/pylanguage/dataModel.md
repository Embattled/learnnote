
# 1. Data Model

## 1.1. Objects, values and types

## 1.2. 标准类型层级结构

## 1.3. Special method names

一个自定义的类可以通过实现一些特殊的函数来实现被特殊操作符调用, 例如 数值计算, 索引, 切片等   
其本质是进行 Python 的操作符重载, 例如 

对于语法 `x[i]`, 在内部被转换为
* `type(x).__getitem__(x,i)`  
* 因此对该类的 `__getitem__` 接口改写可以实现自定义的索引效果

此外, 将特殊函数重载为 None, 代表了对应的操作在该类上不可用
* 例如将 `__iter__` 设为 None, 则该类不可迭代

### 1.3.1. Basic customization

基础的对象行为  

`object.__new__(cls[, ...])`  : 创建类 cls 的一个新的实例, 该接口是一个类的静态函数
* 第一个参数需要传入要创建的类的本身
* 其余参数会传入类的构造函数
* 通常情况下手动重写该函数时, 需要调用带有适当参数的 superclass 的 `__new__`, 即 `super().__new__(cls[,...])`, 然后再进行定制化的修改
* 与 `__init__` 的关系
  * 如果类的 new 接口返回类的实例, 且构造函数里调用 `__new__`. 那么实例的 `__init__` 则会被调用  
  * 如果 new 接口 不返回类的 实例, 那么 实例的 init 则不会被调用
* `__new__` 的存在主旨是为了 那些继承了不可变类型 (int,str,tuple) 的子类可以自定义实例创建.


`object.__init__(self[, ...])` : 类的构造函数
* 在实例被 `__new__` 创建后, 创建的实例被返回前 调用
* 参数就是传递给类构造函数的参数
* 如果基类有自己的 `__init__`, 那么其派生子类必须显式的调用基类的 init, 通过 `super().__init__([args])`
* init和 new 是协同工作的, new 创建对象, init 自定义对象, 因此 `__init__` 接口必须返回 None




### 1.3.2. Customizing attribute access

