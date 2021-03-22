# 1. Numeric and Mathematical Modules

Python 也提供了与数字和数学相关的一系列标准库  

* `random` 提供了常用的伪随机数生成功能
* `numbers` 定义了抽象的分级数值类型
* `cmath math` 包括了多种数学运算, 包括浮点数和复数
* `decimal` 提供了十进制数的拓展表示方法


# 2. random — Generate pseudo-random numbers

该模组提供了伪随机数的生成器  


## 2.1. 生成器


### 2.1.1. 整数生成


*  random.randrange(stop)
*  random.randrange(start, stop[, step])
从参数指定的 range 里随机选一个整数返回, 参数的输入等同于对 range 的输入, 但是并不会真的创建一个range  

*  random.randint(a, b)
从 a到b 的数中选一个整数返回, 相当于`randrange(a, b+1)`  

*  random.getrandbits(k)
返回一个由 k 个随机bit 生成的非负整数

### 2.1.2. 
