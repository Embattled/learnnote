# 1. Numeric and Mathematical Modules

Python 也提供了与数字和数学相关的一系列标准库  

* `random` 提供了常用的伪随机数生成功能
* `numbers` 定义了抽象的分级数值类型
* `cmath math` 包括了多种数学运算, 包括浮点数和复数
* `decimal` 提供了十进制数的拓展表示方法


# 2. random — Generate pseudo-random numbers

* 该模组提供了伪随机数的生成器  
* 因为是内部自带的 lib, 有必要了解一下



## 2.1. 整数生成 Functions for integers

*  `random.randrange(stop)`
*  `random.randrange(start, stop[, step])`
   * 从参数指定的 range 里随机选一个整数返回
   * 参数等于对 range 的输入, 但是并不会真的创建一个range  

*  `random.randint(a, b)`
   * 从 a到b 的数中选一个整数返回
   * 相当于`randrange(a, b+1)`  

*  `random.getrandbits(k)`
   *  返回一个由 k 个随机bit 生成的非负整数

## 2.2. 序列生成 Functions for sequences

* 用于对序列进行随机 sample
* 


## 2.3. 种子操作 Bookkeeping functions

* 种子操作使得一些运行可以复现
* `random.choice(seq)`
* `random.choices(population, weights=None, *, cum_weights=None, k=1)`


* `random.sample(population, k, *, counts=None)`
  * 从 population 中选 k 个元素作为新 list 返回
  * counts 是 3.9 新加的参数, 可以手动指定 population 中的元素的重复个数
    * `sample(['a', 'b'], counts=[4, 2], k=5)` is equivalent to 
    * `sample(['a', 'a', 'a', 'a', 'b', 'b'], k=5)`.