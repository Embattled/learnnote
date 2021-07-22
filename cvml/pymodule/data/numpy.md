# 1. numpy 

* python数值包 最基础的列表处理包 被其他许多包所依赖  
* python stl 中的 math 有许多同名函数, 但不支持向量输入, 因此机器学习中更多的使用 numpy

# 2. numpy.array

NumPy’s array class is called `ndarray`. It is also known by the alias `array`.  
  
* numpy最基础的类, 所有元素都是同一类型, nd 代表可以是任意维度
* 和 python 自带的 `array.array` 是不同的  
* python 自带的类只能处理一维数组, 并且函数很少

`A = np.array([[1,3,4], [2,3,5], [1,2,3], [5,4,6]])`  
注意 ndarry 的打印特点, 值之间没有逗号, 而是空格

## 2.1. attributes

这些都是属性, 不是方法, 调取不加括号
* ndim      : 维度
* shape     : 元组形式表示各个维度的大小, 等同于 tensor.size()
* size      : 总共的元素个数
* dtype     : 元素类型
* itemsize  : 元素大小 bytes
* data      : 指向实际数据的 `array` 一般不需要使用

## 2.2. 元素运算

1. 基础数学运算都是元素为单位的, 一个新的 array 会生成并返回
   * 加减乘除, 乘法会进行元素乘
   * 指数运算`**`
   * 大小判断会返回一个只有 布尔类型的 array
2. 矩阵乘法
   * elementwise product   `A * B`
   * 使用矩阵乘法可以使用    `@` 运算符
   * matrix product        `A @ B`
   * 使用矩阵乘法也可以用    .dot() 方法
   * another matrix product`A.dot(B)`

两个类型不同的 np.array 运算, 结果元素类型会是更通用的或者更精确的一方  
numpy的二元基础运算都是元素层面的  
* `+ - ** *`
* `*`乘法也是元素层面, 两个矩阵使用 `*`  不会进行矩阵乘法
* 组合运算符 `+= -=` 会进行直接结果替换

### 2.2.1. universal function

* 基础运算之外, 在numpy 包中有一些其他的数学运算, 也是元素层面的
* 也有基础运算的全局函数版
* `np.sin(A) np.exp(A)  np.add(A,B)`


## 2.3. Calculation 降维运算

对元素进行某种会带来降维的运算操作, 也可以指定维度 

基本上都有类方法和全局方法两种实现
1. `ndarray.fun(*)`
2. `numpy.fun(array,)`

全局通用参数
* `axis=None` None or int or tuple of ints, optional
   通过指定函数的 `axis=i` 参数, 可以指定运算所执行的对象
    * `i` 从0开始, 指代最外层的括号, 即索引访问时最左边的方括号
    * 其他维度的索引每取一个值, 对 所有 i 维度的值进行运算
* `dtype` 指定, 来保证运算时不会溢出, 默认是相同类型


### 数学类
* `ndarray.max()  numpy.amax()`
* `ndarray.min()  numpy.amin()`
* `ndarray.argmax() numpy.argmax()` 返回最大值对应的索引
* .sum() 返回全元素的和
* .cumsum() 累加该 array


### 2.3.2. 布尔类

* numpy.all 是否全部为 True
* numpy.any 是否有 True

## 2.4. creation

### 2.4.1. numpy.array()

从既存的序列或者元组来创建  
`numpy.array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0, like=None)`
  * 必须以序列的形式传入第一个参数, 即用方括号包住
  * 元素类型会被自动推导, 或者用`dtype=`指定类型

### 2.4.2. 生成函数

1. 使用基础矩阵生成函数, 默认是 float64, 也可以指定类型
   * one()
   * zero()
   * empty()  不确定值, 取决于该块内存原本的值
   * random() 
   * 第一参数为元组, 指定维度大小
2. 使用 `arange` 生成等差整数序列
   * 使用方法基本和 内置函数 range 相同 (起始值, 结束值, 间隔)
   * python 内置函数 range 返回的是普通 array, 该方法返回的是 np.array
3. 使用 `linspace` 创建浮点数序列
   * 同 arange 相同, 只不过思想改成了分割
   * linspace(起始值,结束值,分割成多少个值)
   * 和 range 不同, 结束值会被包括进去

```py
a = np.arange(15).reshape(3, 5)
""" 
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
"""


```

## 2.5. 形态转换 Shape Manipulation

a.shape 可以返回当前的 array 的形状

函数可以分为
1. 类方法和全局函数
2. 返回改变后的值和改变自身


以下函数都有 类方法和全局函数, 类方法没有第一个参数 array
1. resize  (a, tuple )  更改这个 array 自身
2. reshape (a, tuple )  返回更改后的 array
3. 如果某个维度的值是 -1, 则该维度的值会根据 array的数据量和其他维度的值推算出来

类方法
* a.T         返回 transposed 的 array

扁平化array
1. numpy.ravel(a)          返回 flattened 的 array 
2. a.flatten()             返回 flattened 的 array

扁平化索引, 不需要用多重方括号索引值
* `a.flat[1]`

### 2.5.1. 拼接 stacking together

* numpy.vstack((a,b)) == numpy.row_stack() 
  * 上下拼接
* numpy.hstack((a,b))
  * 左右拼接
* numpy.column_stack((a,b))
  * 将 1D array 视作列进行左右拼接
  * 在处理 2D array 时与 hstack 相同

### 2.5.2. 拆分 splitting

* numpy.hsplit(a,3) 竖着切3分
* numpy.vsplit(a,3) 横着切, 沿着 vertical axis
* numpy.array_split 指定切的方向

### 升维 

1. `arr=arr[:,:,np.newaxis`
2. `arr=np.array([arr])`

### 降维 


* `np.squeeze(x)`
* 删掉维度数为1的轴
* Remove axes of length one from a.

```py
>>> x = np.array([[[0], [1], [2]]])
>>> x.shape
(1, 3, 1)
>>> np.squeeze(x).shape
(3,)
>>> np.squeeze(x, axis=(2,)).shape
(1, 3)
```

## 2.6. Copies and Views

* 普通赋值不进行拷贝       `a=b`
* view不进行数据拷贝       `c=a.view()` 算是绑定了一个引用, 可以用来操纵数据
* copy方法会进行完整拷贝   `d=a.copy()`

对于截取数据中的有用片段, 删除无用片段时, 需要进行拷贝
```py
a = np.arange(int(1e8))
b = a[:100].copy()
del a  # the memory of ``a`` can be released.
```



# 3. numpy.random

* numpy 的随机包是独立出来的 `numpy.random`
* 比 Pystl 的 random 包通用性更广, 值得学习

基本概念
1. BitGenerators: 生成随机数的对象, 由随机32 or 64 bit 填入的 unsigned interger 
2. random generator: 主要负责将随机bit转为特定分布
3. Generators: 该库的用户接口, 将 BitGenerator 的随机序列转换为特定的概率分布以及范围的随机数
4. RandomState: 已经被淘汰的旧版生成器, 只能基于一种 BitGenerators
   - RandomState 目前直接作为一个对象被放入了 该命名空间

## 3.1. Generator

基本生成器
* `default_rng(seed=None)`  : 被官方推荐的生成器, 使用 `PCG64` , 效果优于 `MT19937`
  - seed : None, int, array_like`[ints]`, SeedSequence, BitGenerator, Generator. 


通过使用生成器对象的方法可以产生任意区间和分布的随机数
* Simple random data 简单随机数
  * `integers(low[, high, size, dtype, endpoint])`
    - 产生整数
    - 默认 dtype= np.int64
  * `random([size, dtype, out])`
    - 可以用来初始化网络, 随机floats in interval [0.0, 1.0).
  * `choice(a[, size, replace, p, axis, shuffle])`
    - 从给定的 a 数组中进行随机采样, 如果 a 是int, 则从 np.arange(a) 中采样
    - size 指定采样的大小, None 代表采样 1 个
    - replace : 是否允许替代
    - p : array, 给定 a 中每个元素的采样概率
  * `bytes(length)`
    - 返回一个 random bytes

* 分布采样, 基于特定分布的采样方法
  * `random.Generator.normal(loc=0.0, scale=1.0, size=None)`
    - loc scale , 代表 mean 和标准差
  * `random.Generator.standard_normal(size=None, dtype=np.float64, out=None)`
    - 从标准的正态分布采样
  * `random.Generator.uniform(low=0.0, high=1.0, size=None)`
    - 均一分布

# 4. config

## 4.1. np.set_printoptions

1. 取消科学计数法显示数据 `np.set_printoptions(suppress=True)  `



2. 取消省略超长数列的数据 ` np.set_printoptions(threshold=sys.maxsize)` 需要 sys 包


### 4.1.1. numpy.shape

### 4.1.2. numpy.dot()  矩阵点乘

np.diag(s)  将数组变成对角矩阵  
使用numpy进行矩阵乘法   

```py
# 使用svp分解矩阵
U, s, Vh = np.linalg.svd(A, full_matrices=False)
#使用 .dot() 将原本的矩阵乘回来 
Us = np.dot(U, np.diag(s))
UsVh = np.dot(Us, Vh)

```


```py

# 计算一个矩阵的奇异值
A = np.array([[1,3,4], [2,3,5], [1,2,3], [5,4,6]])
U, s, Vh = np.linalg.svd(A, full_matrices=False)

print(np.shape(U), np.shape(s), np.shape(Vh))

'''输出
(4, 3) (3,) (3, 3)
'''
```




## 4.2. linalg 

### 4.2.1. SVD 奇异值分解

Singular Value Decomposition  
* M = U * s * Vh  
* U  : Contains all the information about the rows (observations)  
* Vh: Contains all the information about the columns (features)  
* s   : Records the SVD process  

奇异值可以用来压缩数据  

```py
# 对原始数据进行SVD分解
A = np.array([[1,3,4], [2,3,5], [1,2,3], [5,4,6]])
U, s, Vh = np.linalg.svd(A, full_matrices=False)

# 删除了不重要的一行/列数据, 再不影响最终维度的情况下,仍能得到原本的矩阵
Us = np.dot(U[:,:2], np.diag(s[:2]))
UsVh = np.dot(Us, Vh[:2,:])
'''
[[1.  2.8 4.1]
 [2.  3.2 4.8]
 [1.  2.  3. ]
 [5.  3.9 6. ]]
 '''
# 若是删除了两行,则不能保证还原数据
Us = np.dot(U[:,:1], np.diag(s[:1]))
UsVh = np.dot(Us, Vh[:1,:])
'''
[[2.1 2.5 3.7]
 [2.6 3.1 4.6]
 [1.6 1.8 2.8]
 [3.7 4.3 6.5]]
'''

# 一般通过统计 s 中的数值占比
s_percentage = (s/sum(s)*100).round(2)



```
As a general rule, you should consider solutions maintaining from 70 to 99 percent of the original information.  
