# Functional Programming Modules

The modules described in this chapter provide functions and classes that support a functional programming style, and general operations on callables.

用于支持 functional programming style 的类


# itertools — Functions creating iterators for efficient looping¶


该模组实现了一些 迭代器构建模块, 许多都是根据 `Haskell` `SML` 启发的
* 主要是实现了 fast, memory efficient tools
* 这些工具可以被单独或者组合使用
* 所构成的概念称为 `iterator algebra`, 使得可以方便的通过 python 构建高效的专用工具

## Itertool functions

itertools 下的函数都是用来构建一个 iterators.  
其中的一些直接提供了无限长度, 因此必须使用带截断的循环来访问  


### Infinite iterators

### Iterators terminating on the shortest input sequence

对于输入的多个 sequence, 在最短的 sequence 上终止 

`itertools.accumulate(iterable[, func, *, initial=None])`
* 默认返回输入序列的累加
* 如果设置了 initial, 则 len(iterator) = len(input) + 1
* 如果传入 func 来指定非默认的累加函数, 则 func 需要是接受两个参数的函数 , 跟多时候可以使用 `operator` 模组里的既定函数  
  * 可以通过 `accumulate(data, operator.mul)` 来间接实现 `product` 函数的功能



### Combinatoric iterators


`itertools.product(*iterables, repeat=1)`
* 返回输入迭代对象的 `Cartesian product`  笛卡尔积
* `product(A, B) returns the same as ((x,y) for x in A for y in B).`
* 如果设置了 repeat, 则会将输入的 iterables 平铺 repeat 次
  * `product(A, repeat=4)` means the same as `product(A, A, A, A).`
* 在运行 product 的时候, 会先消耗所有的输入并存储在内存中, 因此不能输入长度可变的iterable, 必须是 finite
  




## operator — Standard operators as functions

python 的运算符所内置的函数体, `operator.add(x, y)` is equivalent to the expression `x+y`.  
* 许多函数都是作为旧版本的兼容函数保留的, 因此命名为了双下划线版本 `__func__`, 在实际使用中最好使用非下划线版本  

大体上, operator function 是可以进行简单分类的
* object comparisons
* logical operations
* mathematical operations
* sequence operations