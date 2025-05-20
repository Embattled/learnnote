# 1. Functional Programming Modules

The modules described in this chapter provide functions and classes that support a functional programming style, and general operations on callables.

用于支持 functional programming style 的类


# 2. itertools — Functions creating iterators for efficient looping¶


该模组实现了一些 迭代器构建模块, 许多都是根据 函数式编程语言 `APL` `Haskell` `SML` 启发的
* 主要是实现了 fast, memory efficient tools
* 这些工具可以被单独或者组合使用

该模组标准化了一个 核心工具 (core set), 提供 fast, memory efficient tools 用于构成
* 称为 `iterator algebra`, 使得可以方便的通过 pure python 构建高效的专用工具

## 2.1. Itertool functions

itertools 下的函数都是用来构建一个 iterators.  
其中的一些直接提供了无限长度, 因此必须使用带截断的循环来访问  


### 2.1.1. Infinite iterators

### 2.1.2. Iterators terminating on the shortest input sequence

对于输入的多个 sequence, 在最短的 sequence 上终止 

`itertools.accumulate(iterable[, func, *, initial=None])`
* 默认返回输入序列的累加
* 如果设置了 initial, 则 len(iterator) = len(input) + 1
* 如果传入 func 来指定非默认的累加函数, 则 func 需要是接受两个参数的函数 , 跟多时候可以使用 `operator` 模组里的既定函数  
  * 可以通过 `accumulate(data, operator.mul)` 来间接实现 `product` 函数的功能



### 2.1.3. Combinatoric iterators


`itertools.product(*iterables, repeat=1)`
* 返回输入迭代对象的 `Cartesian product`  笛卡尔积
* `product(A, B) returns the same as ((x,y) for x in A for y in B).`
* 如果设置了 repeat, 则会将输入的 iterables 平铺 repeat 次
  * `product(A, repeat=4)` means the same as `product(A, A, A, A).`
* 在运行 product 的时候, 会先消耗所有的输入并存储在内存中, 因此不能输入长度可变的iterable, 必须是 finite
  

## 2.2. Itertools Recipes

# 3. functools — Higher-order functions and operations on callable objects

higher-order functions: functions that act on or return other functions. 作用于函数 或者 返回值是一个函数的 函数.

一般来说, 所有的 python callable object 可以被作为 function 来应用该模组

用于便捷的将函数封装成特殊形式的库
该库中所有对象都是函数装饰器  

* cache       : 函数调用获得 unbounded function cache
* cached_property
* cmp_to_key
* lru_cache   : 函数调用获得指定数量的缓存 (least recently used cache)
* total_ordering
* partial
* partialmethod
* reduce
* singledispatch
* singledispatchmethod
* `functools.update_wrapper(wrapper, wrapped, assigned=WRAPPER_ASSIGNMENTS, updated=WRAPPER_UPDATES)`
* `@functools.wraps(wrapped, assigned=WRAPPER_ASSIGNMENTS, updated=WRAPPER_UPDATES)`

## 3.1. 函数缓存 cache

`@functools.cache(user_function)` : Simple lightweight unbounded function cache. Sometimes called `memoize`.
* 函数会通过一个 dict 保存所有调用的函数值, 且永远不逐出旧值, 对于重复的调用会直接从 dict 获取值
* 同 `lru_cache(maxsize=None)` 的行为相同
* cache 缓存是线程安全的, 所以函数本身可以应用在多线程上. 线程之间的 cache dict 会保持内容更新.
* 对于极端情况, 首次函数调用结束之前就在其他线程发起了另一个调用, 则目标函数可能会被执行多次.
* python 3.9 的新内容
```py
@cache
def factorial(n):
    return n * factorial(n-1) if n else 1
```


`@functools.lru_cache(user_function)`
`@functools.lru_cache(maxsize=128, typed=False)`  : 将函数封装为可记忆的函数, 保存最近 maxsize 次的调用
* 用途, 对于周期性以相同参数调用的, 计算昂贵 or I/O 绑定的函数的时候, 可以节省时间
* cache 是 threadsafe , 所以可以在多个线程中安全使用, 同时保持底层数据接口在并发更新期间保持一致
* 同 `cache` 一样, 如果在首次调用返回之前就有并发的调用, 则相同的函数可能会被调用多次
* `lru_cache` 使用字典来存储调用的返回值, 参数会作为字典的 key, 所以参数必须是 `hashable` 的
  * 关键字参数的顺序也会作为 key 来保存, 因此即使只有顺序不同, 也会被作为单独的缓存条目
  * `maxsize` 传入 None 即可使 cache 大小无限

```py
@lru_cache
def count_vowels(sentence):
    return sum(sentence.count(vowel) for vowel in 'AEIOUaeiou')
```

## 3.2. 类方法描述符

* cached_property   : 将某个方法封装为属性

### 3.2.1. 属性缓存 


`@functools.cached_property(func)`    : 将一个 method 转换为 property
* 该方法只会被计算一次, 然后其值会被缓存并作为类的属性 for the life of the instance
* 从行为上来说和 property() 类似, 但是增加了缓存
  * property() 会阻止属性通过 setter 以外的方法写入
  * cached_property 允许直接写入属性
* 对于计算量很大的 实例属性来说很有用
  * 计算只会在查找对应属性的时候运行, 且仅在不存在同名属性的时候运行
  * 运行时 cached_property 会写入同名的属性, 且后续对于该 property 的操作同一般的 property 操作方法相同
  * 无论是读取还是写入, cached 是属性值的优先级都高于被 cached_property 标记的方法
* 对于多线程, 该描述符不会进行任何特殊对应  
  * 对应的 getter 方法会获取对后一次运行的 cached value
  * 如果能确保该描述符对应的方法是 idempotent 或者没有有害影响, 则无所谓, 否则需要用户自己定义并发执行时候的锁
* 更新:
  * 于 3.8 加入
  * 在 3.12 更新. 删除了一个 文档中没有说明的锁机制. 
* 使用限制: 该描述符需要实例拥有 `__dict__` 属性 且是 `mutable mapping`  
  * 该限制导致某些类型无法应用该描述符, 例如 `metaclasses`. metaclasses 的实例的 `__dict__` 属性是 read-only proxies
  * 以及对于 `__slots__` 属性不包括 `__dict__` 作为已定义插槽之一 的类型, 例如不提供 `__dict__` 属性的类
* 该描述符会干扰 `PEP412` key-sharing dictionaries 的操作, 这会导致实例字典会占用更多储存空间
* 对于 mutable mapping 不可用 或者空间敏感的需要 key sharing 字典的应用场景, 可以用过结合 `property()` 和 `lru_cache()` 来实现类似的功能, 在具体内部实现上的区别可以参照别的文章
  [faq: How do I cache method calls?](https://docs.python.org/3/faq/programming.html#faq-cache-method-calls)

```py
class DataSet:
    def __init__(self, sequence_of_numbers):
        self._data = tuple(sequence_of_numbers)

    @cached_property
    def stdev(self):
        return statistics.stdev(self._data)
```


## partial - 固定参数


`functools.partial(func, /, *args, **keywords)`
* 创建一个新的可调用的对象, 该对象直接调用的时候等同于 使用 args 和 kwargs调用原始对象
* 新对象的位置参数会 appends, kwargs 则会扩展并且覆盖
* partial 的功能可以简单的理解为 冻结 `freezes` 一部分参数, 使得函数调用拥有更加简洁的形式  


```py
# partial 的简易实现原理等同于
def partial(func, /, *args, **keywords):
    def newfunc(*fargs, **fkeywords):
        newkeywords = {**keywords, **fkeywords}
        return func(*args, *fargs, **newkeywords)
    newfunc.func = func
    newfunc.args = args
    newfunc.keywords = keywords
    return newfunc

# 创建二进制 str 转为 int 的 partial
from functools import partial
basetwo = partial(int, base=2)
basetwo.__doc__ = 'Convert base 2 string to an int.'
basetwo('10010')
# 18
```


`class functools.partialmethod(func, /, *args, **keywords)`
* 动态创建一个 partial, 拥有类似的功能
  * 区别在于 used as a method definition rather than being directly callable
  * 主要用于为类的方法做修饰
* 参数 `func` 必须是 descriptor or callable
  * (objects which are both, like normal functions, are handled as descriptors).
  * 如果 func 是 descriptor
    * 对 `__get__` 的调用会被委托给底层的描述器, 并会返回一个适当的 部分对象 作为结果 (看不懂)
  * 如果 func 不是 descriptor 的 callable 的时候
    * 动态创建一个适当的绑定方法
    * 且使用的时候类似普通函数
    * 将会插入 self 参数作为第一个位置参数, self 会自动在所有 args 之前

```py

class Cell:
    def __init__(self):
        self._alive = False

    @property
    def alive(self):
        return self._alive

    def set_state(self, state):
        self._alive = bool(state)

    set_alive = partialmethod(set_state, True)
    set_dead = partialmethod(set_state, False)


c = Cell()
c.alive
# False

c.set_alive()
c.alive
# True
```

## 3.3. wraps - 封装函数修饰器

`functools.update_wrapper(wrapper, wrapped, assigned=WRAPPER_ASSIGNMENTS, updated=WRAPPER_UPDATES)`
* 使一个封装函数 看起来就像 被他封装的原始函数一样, 如果不这么做的话, 封装函数无法访问任何函数的有用属性, 例如 文档, 参数等, 会大大影响封装函数的易用性
* 具有两个可选参数, 类型都是 tuple
  * `assigned` : 指定被封装的原始函数 的哪些属性要直接赋值给 封装函数的相对应匹配属性
  * `updated`  : 指定封装函数的哪些属性要使用 原函数的相应属性来更新
  * 不太懂都是从原始函数 -> 封装函数, 为啥要区分 赋值和更新: 因为有一个 dict 的属性, 使用更新更好
  * 默认值都是模块级常量
    *  `WRAPPER_ASSIGNMENTS` : 赋值部分, 包括 `__module__, __name__, __qualname__, __annotations__, __type_params__, __doc__`
    *  `WRAPPER_UPDATES `    : 更新部分, 包括 `__dict__`
* 也就是说基本上所有属性都会直接继承原本的函数
  * annotations 是松散的字典
  * wraps 不会影响 函数的签名, 即 IDE 会使用的 `inspect.signature()`, 参考 typing 的 ParamSpec 
* 对于某些高级用途, 需要直接访问原始函数的情况下: `introspection` 或者 `bypassing a caching decorator such as lru_cache()`
  * 该函数会给 封装函数添加一个 `__wrapped__ ` 用于指向原始的函数  
*  更新历史
   *  3.2 : `__wrapped__` 属性会被自动添加, `__annotations__` 属性会被拷贝, 原始函数如果缺少对应属性的话不会报错
   *  3.4 : `__wrapped__ ` 属性总是指向原始函数, 即使原始函数本身也是一个封装函数, 具有 `__wrapped__` 属性
   *  3.12: `__type_params__` 会被默认拷贝
 


`@functools.wraps(wrapped, assigned=WRAPPER_ASSIGNMENTS, updated=WRAPPER_UPDATES)`
* update_wrapper 的便捷调用方法
* 等同于 `partial(update_wrapper, wrapped=wrapped, assigned=assigned, updated=updated)`


# 4. operator — Standard operators as functions

python 的运算符所内置的函数体, `operator.add(x, y)` is equivalent to the expression `x+y`.  
* 许多函数都是作为旧版本的兼容函数保留的, 因此命名为了双下划线版本 `__func__`, 在实际使用中最好使用非下划线版本  

大体上, operator function 是可以进行简单分类的
* object comparisons
* logical operations
* mathematical operations
* sequence operations