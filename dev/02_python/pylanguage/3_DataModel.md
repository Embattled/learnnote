
# 1. Data Model

## 1.1. Objects, values and types - 对象, 值, 类型 的定义
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





## 1.2. The standard type hierarchy - 标准类型层级结构

在本章节会介绍 Python 的内置类型.   
扩展模块可以定义其他类型


## 1.3. Special method names
<!-- 头部完 -->
一个自定义的类可以通过实现一些特殊的函数来实现被特殊操作符调用, 例如 数值计算, 索引, 切片等   
其本质是进行 Python 的操作符重载, 例如 

对于语法 `x[i]`, 在内部被转换为
* `type(x).__getitem__(x,i)`  
* 因此对该类的 `__getitem__` 接口改写可以实现自定义的索引效果

此外, 将特殊函数重载为 None, 代表了对应的操作在该类上不可用
* 例如将 `__iter__` 设为 None, 则该类不可迭代


一个 class 可以通过实现特殊名称的 方法, 来实现由特殊语法调用的某些操作. 即运算符重载, 索引, 切片 等
例如 `__getitem__()` 方法可以让类能够进行索引操作  
同时, 如果运算符并未有对应操作的对应方法, 则会唤起 ERROR, 通常是  AttributeError or TypeError
手动将某个特殊名称的方法设置为 None, 表示对应类不可以进行相应操作, 可以避免默认行为导致的歧义


### 1.3.1. Basic customization - 基础的行为自定义

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


`object.__repr__(self)` : 用于定义该 对象的 `正式 offical` 字符串表达形式, 会作为 `repr(obj)` 的返回值
* 返回值必须是 字符串对象
* 如果有条件的话, 返回值应当可以被用于 重新创建 一个具有相同值的对象 `的 Python 表达式`
* 如果上述条件不满足的话, 则最起码应该返回一个  `<...some useful description...>` 样子的字符串
  * 可以参考在 交互式终端中输出 np.ndarray 时候的形式
* 对于 `非正式 informal` 字符串形式的 `__str__` 如果没有被定义的话, `__repr__` 则会被替代使用
* 该方法通常用于 debug, 因此输出信息的量往往非常丰富且明确


`object.__str__(self)`   : 用于该对象的 `informal` or `nicely printable` 字符串表达, 会作为 `str(obj)` 的返回值, 用于 `format() print()`
* 返回值必须是字符串对象
* 不期望返回值是一个 Python 表达式
* 该函数的默认实现是直接调用 `__repr__()`



### 1.3.2. Customizing attribute access



### With Statement Context Manager

context manager : 上下文管理器
* 通常当执行 with 表达式的时候被创建, 也可以直接被调用
* 上下文管理器用于处理 entry into 和 exit from

通常 context managers 用于暂时的保管以及回复某些全局状态, locking and unlocking resources, 打开即关闭文件等  

python 的基本类 object 没有实现上下文管理器的方法  

`object.__enter__(self)`
* 用于执行 runtime context 启动的时候的语句
* with 表达式会 bind this method's return value to the target(s) specified in the `as` clause of the statement

`object.__exit__(self, exc_type, exc_value, traceback)`
* 推出 context manager 时候执行的方法
* 该方法的三个参数都是用于出现异常时候的退出的, 如果程序没有异常, 则三个参数都是None
* 如果希望 context manager 抑制传入的异常, 则该方法需要返回 True.
* `__exit__` 方法本身不应该重复唤起传入的异常



