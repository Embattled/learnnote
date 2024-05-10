# 1. Functional Programming Modules

The modules described in this chapter provide functions and classes that support a functional programming style, and general operations on callables.

用于支持 functional programming style 的类


# 2. itertools — Functions creating iterators for efficient looping¶


该模组实现了一些 迭代器构建模块, 许多都是根据 `Haskell` `SML` 启发的
* 主要是实现了 fast, memory efficient tools
* 这些工具可以被单独或者组合使用
* 所构成的概念称为 `iterator algebra`, 使得可以方便的通过 python 构建高效的专用工具

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
  



# 3. functools — Higher-order functions and operations on callable objects

higher-order functions: functions that act on or return other functions. 作用于函数 或者 返回值是一个函数的 函数.

一般来说, 所有的 python callable object 可以被作为 function 来应用该模组

## 3.1. 函数描述符

用于便捷的将函数封装成特殊形式的库

* cache       : 函数调用获得 unbounded function cache
* lru_cache   : 函数调用获得指定数量的缓存 (least recently used cache)

### 3.1.1. 函数缓存 cache

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


# 4. operator — Standard operators as functions

python 的运算符所内置的函数体, `operator.add(x, y)` is equivalent to the expression `x+y`.  
* 许多函数都是作为旧版本的兼容函数保留的, 因此命名为了双下划线版本 `__func__`, 在实际使用中最好使用非下划线版本  

大体上, operator function 是可以进行简单分类的
* object comparisons
* logical operations
* mathematical operations
* sequence operations