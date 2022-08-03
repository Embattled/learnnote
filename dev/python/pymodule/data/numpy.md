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

## 2.2. scalars 

Python 为了便于书写, 对于整数和小数, 各自只定义了一种数据类型: integer float  

numpy 由于需要应用在科学运算上, 数据的类型变得十分重要
* numpy 定义了总计 24 种数据类型  
* 这些数据类型主要基于 C 语言以及 Cpython
* 还有一些数据类型用于保持与 python 内建类型的兼容性



## 2.3. 元素运算

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

### 2.3.1. universal function

* 基础运算之外, 在numpy 包中有一些其他的数学运算, 也是元素层面的
* 也有基础运算的全局函数版
* `np.sin(A) np.exp(A)  np.add(A,B)`

### 2.3.2. clip 修正边界

* numpy.clip(a, a_min, a_max, out=None, **kwargs) 
* 将 array 限制在正确的范围里


## 2.4. Calculation 降维运算

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


### 2.4.1. 数学类
* `ndarray.max()  numpy.amax()`
* `ndarray.min()  numpy.amin()`
* `ndarray.argmax() numpy.argmax()` 返回最大值对应的索引
* .sum() 返回全元素的和
* .cumsum() 累加该 array




## 2.5. creation

### 2.5.1. numpy.array()

从既存的序列或者元组来创建  
`numpy.array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0, like=None)`
  * 必须以序列的形式传入第一个参数, 即用方括号包住
  * 元素类型会被自动推导, 或者用`dtype=`指定类型

### 2.5.2. 生成函数

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


## 2.7. Indexing routines 

有很多包函数没有放在 numpy routines, 而是放在该部分, 以后再读   


# 3. Routines 常规操作 API

对 Array 数据的各种常规操作

* 尽量明确各个函数的返回值是 copy 的还是只是映射

函数可以分为
1. 类方法和全局函数
2. 返回改变后的值和改变自身(inplace)


## 3.1. Array creation

numpy.array 的各种创建函数
* 所有函数都有 `dtype` 参数
* 所有函数都有 `order` 参数
* 部分函数里有 `like`  参数 : array_like, optional, 用于将返回值创建成 np.array 以外的数据类型


### From shape or value

基础创建函数, 需要指定 array 的 shape
* `*_like` 版本函数, 输入的不再是 shape 而是另一个 array, 相当于 `func(a.shape,...)` 的另一种写法
  * 该版本创建的 array 默认数据类型同原本的 array 一致
* `ones(shape[, dtype, order, like])`
* `ones_like(a[, dtype, order, subok, shape])`
* `zeros(shape[, dtype, order, like])`
* `zeros_like(a[, dtype, order, subok, shape])`
* `full(shape, fill_value[, dtype, order, like])`
* `full_like(a, fill_value[, dtype, order, ...])`
* `empty(shape[, dtype, order, like])`  不初始化对应内存区域
* `empty_like(prototype[, dtype, order, subok, ...])`

特殊 array
* `identity(n[, dtype, like])`    Return the identity array. 只能是2D正方形, 所以传入的 shape 是单个整数
* `eye(N[, M, k, dtype, order, like])`  创建2d对角线矩阵, 传入的 shape 可以是长方形, k 代表对角线的偏移


### From existing data

从既存的数据中创建一个 array


`numpy.fromfile(file, dtype=float, count=- 1, sep='', offset=0, *, like=None)`  文件读取
* `file`  : Open file object or str or Path, 1.17.0 pathlib.Path objects are now accepted.
* `dtype` : 用于指定 array 类型, 同时对于 binary 文件还决定了读取的步长 (即单个 item 的大小)
* `count` : 用于指定要读取的 item 的数量, -1代表读完整个文件
* `sep`   : 指定了该文件是否是 binary 或者 text file, 默认是空字符代表了二进制文件, 如果是空格分隔符 ` `, 则代表 text 文件, 同时分割匹配符会匹配1到多个空白字符
* `offset`: 读取时候的向后偏移, 只在 binary 的时候起作用



## 3.2. Array manipulation 操纵更改 Array

获取 array 的形态
* `a.shape`
* `np.shape(a)`

拷贝array
* `numpy.copyto(dst, src, casting='same_kind', where=True)`  
  * 把一个 array 的值复制到另一个 array
  * 不同于普通的 copy, 这里 dst 也是提前存在的, 会进行 broadcasting
  * where : array of bool, 附加的元素选择
  * casting : cast模式, 在别的地方应该能学到此处略

### 3.2.1. Transpose-like operations 转置操作

普通转置
* `a.T`
* `a.transpose(*axes)`
  * 返回 view of the array with axes transposed.
* `numpy.transpose(a, axes=None)`
  * 返回 A view is returned whenever possible.
* `axes` 默认是 `range(a.ndim)[::-1]` 用于完整颠倒 order of the axes, 如果指定的话则必须是permutation of `[0,1,..,N-1]` 

### 3.2.2. Changing array shape 形态转换

除了转置以外的其他各种形态操作  

标准形态转换 reshape
* `numpy.reshape(a, newshape, order='C'`  
* `ndarray.reshape(shape, order='C')`
* newshape : int or tuple of ints
  * 如果是单个整数, 则 1-D array of that length
  * 如果是元组, 则 The new shape should be compatible with the original shape. 
  * 元组的情况下, 允许有一个维度为 -1, 此时该维度会自动被计算.
* order: 
  * 内存布局
  * C : C语言布局, 末尾的维度变化最快, 内存连续
  * F : Fortran-like, 首位维度变化最快
  * A : 自动根据内存布局选择
* 返回新的 array:
  *  will be a new view object if possible
  *  otherwise, will be a copy 


扁平化array
1. `numpy.ravel(a, order='C')`
  * Return a contiguous flattened array.
  * A copy is made only if needed.
  * 返回值的 shape 等同于 a.size
2. `a.flatten(order='C')`
  * 返回 flattened 的 array
  * copy 的返回 
3. a.flat  : 不是函数, 而是一个 iterator 对象
  * 扁平化索引, 不需要用多重方括号索引值
  * `a.flat[1]`


### 3.2.3. Changing dimensions 维度操作




#### 3.2.3.1. expand_dims 升维

朴素升降维:
* `numpy.expand_dims(a, axis)`
  * axis 的范围是 (0,a.ndim)
  * 被插入的维度的位置 == axis , 意思是其后的维度会被顺延
  * return : `View` of a with the number of dimensions increased.

1. `arr=arr[:,:,np.newaxis`
2. `arr=np.array([arr])`


#### 3.2.3.2. squeeze 压缩维度 
* `numpy.squeeze(a, axis=None)`
  * 删掉 shape 为 1 的维度
  * axis 可以指定, 但是指定的维度必须确保 shape == 1
  * return : This is always a itself or a `view` into a. 
    * Note that if all axes are squeezed, the result is a 0d array and not a scalar.

```py
>>> x = np.array([[[0], [1], [2]]])
>>> x.shape
(1, 3, 1)
>>> np.squeeze(x).shape
(3,)
>>> np.squeeze(x, axis=(2,)).shape
(1, 3)
```

### 3.2.4. Joining arrays 拼接

```py
numpy.concatenate((a1, a2, ...), axis=0, out=None, dtype=None, casting="same_kind")
```
`numpy.concatenate()` 指定维度拼接:
* all the input arrays must have same `number of dimensions`
* tup   : 除了 `axis` 指定的维度, 其他维度的shape必须相同
* 注意 ndim 必须相同, 也就是一般来说需要先扩张维度
* axis  : 要拼接的维度, 默认是0 (最外层)


简单拼接
* `numpy.stack(arrays, axis=0, out=None)`       直接在新维度拼接, 该方法不需要提前扩充维度
* `numpy.vstack(tup)` 上下拼接
* `numpy.row_stack(tup)` == vstack
* `numpy.hstack(tup)` 左右拼接
* `numpy.column_stack(tup)`
  * 将 1D array 视作列进行左右拼接
  * 在处理 2D array 时与 hstack 相同

### 3.2.5. Splitting arrays 拆分 

* numpy.hsplit(a,3) 竖着切3分
* numpy.vsplit(a,3) 横着切, 沿着 vertical axis
* numpy.array_split 指定切的方向



### 3.2.6. Adding and removing elements 修改元素

将 np.ndarray 以类似于普通 list 的视角操作

### 3.2.7. append

该函数不存在 in-place 模式

* `numpy.append(arr, values, axis=None)` 
  * arr : Values are appended to a `copy` of this array.
  * values: array_like. 维度 ndim 必须和 arr 一样
  * axis : None 要被插入的维度
    * values : 必须是 correct shape (the same shape as arr, excluding axis).
    * if axis = None, values can be any shape and will be flattened before use
    * If axis is None, out is a flattened array.
  * return : A copy of arr with values appended to axis. 
  

#### 3.2.7.1. resize 强行更改 shape

不同于 reshape 的resize
* 会带有填充以及裁剪的更改 array 形态
* 以 C-order进行
* `numpy.resize(a, new_shape)` : 不足的地方用 a 的重复来填充, 返回新的 array
* `ndarray.resize(new_shape, refcheck=True)` in-place 的修改, 不足的地方会用 0 填充
  * 如果 refcheck 是True, 那么如果该 array 有别的引用的话会发生 raise 来停止该操作
* 



## Logic functions 逻辑操作

包括 ndarray 之间的逻辑运算以及自身元素的检查逻辑

### Truth value testing

只有两个函数
* `numpy.all(a, axis=None, out=None, keepdims=<no value>, *, where=<no value>)` 是否全部为 True
* `numpy.any(a, axis=None, out=None, keepdims=<no value>, *, where=<no value>)` 是否有 True

## 3.3. Mathematical function 数学操作

绝大多数常用函数都属于该分类

### 3.3.1. Trigonometric functions 三角函数

### 3.3.2. Exponents and logarithms 指数

### 3.3.3. Rounding 最近值

### 3.3.4. Rational routines 最大公因数 最小公倍数

### 3.3.5. Extrema Finding 极值寻找

### 3.3.6. 杂项

有时候需要的很特殊的功能

#### 3.3.6.1. convolve 卷积

Returns the discrete, linear convolution of two one-dimensional sequences.
* 常用在信号处理中
* 一维卷积

`numpy.convolve(a, v, mode='full')`  
* a, v 是两个一维array , 需要 a 比 v 长, 否则会在计算前交换顺序
* mode : str
  * full :  the convolution at each point of overlap, with an output shape of (N+M-1).
    * 相当于从左端只有一个元素重合开始, 向右平移 v 直到再次只有一个元素重合 
    * 得到 N+M-1 个值
  * same : returns output of length max(M, N). 
    * 相当于从左端对齐开始, 向右平移 v 直到只有1个元素overlap
    * 得到 N 个值
  * valid: 
    * 没有任何边际值
    * 只计算 a 和 v 完全 overlap 的卷积
    * 得到   max(M, N) - min(M, N) + 1. 个值


## 3.4. Sorting, Searching, Counting 排序 搜索 计数

这里的 counting 都是很简单的函数, 更详细的统计在 statistics 模块


### 3.4.1. Sorting 排序

#### 3.4.1.1. 基础排序

* `msort(a)` : Return a copy of an array sorted along the first axis.


* `sort(a[, axis, kind, order])`      : Return a sorted copy of an array.
* `ndarray.sort([axis, kind, order])` : Sort an array in-place.
* `argsort(a[, axis, kind, order])`   : Returns the indices that would sort an array.

参数:
* axis    : (default -1) int or None, optional
  * None, 全局排序, 会自动将数据 flatten
  * int , 按照指定维度的数据排序, 默认值 -1 代表按照最后一个维度排序
* kind    : {‘quicksort’, ‘mergesort’, ‘heapsort’, ‘stable’}, optional
  * 排序方法
  * default : None  = quicksort
* order   : str or list of str, optional
  * 主要用于结构体的 array
  * 用于通过 结构体字段的名称或者名称list来指定排序比较的顺序


#### 3.4.1.2. 部分有序

* partition(a, kth[, axis, kind, order])    :  Return a partitioned copy of an array.
* argpartition(a, kth[, axis, kind, order]) : 

### 3.4.2. Searching 元素查找

大概可以分成
* 极值查找
  * argmax
  * argmin
  * 
* 逻辑查找
  * where
* 非零查找
  * argwhere
  * nonzero
  * flatnonzero

#### 3.4.2.1. 最大值选择

* `argmax(a[, axis, out, keepdims])`
  * Returns the indices of the maximum values along an axis.
  * 一般用于机器学习通过可信度得到最终 label
* `argmin(a[, axis, out, keepdims])`
  * Returns the indices of the minimum values along an axis.



#### 3.4.2.2. 逻辑选择值 where

`numpy.extract(condition, arr)`
* 根据 condition 选择元素, 等同于 
  * `np.compress(ravel(condition), ravel(arr))`
  * 如果 condition is an array of boolean 则等同于`arr[condition]`
* 完全的与 `numpy.place` 功能相反的函数
* 返回 : 所提取的元素组成的 1-D array



`numpy.where(condition, [x, y, ]/)`
* 用于根据某个判断语句 在两个 array 之间选择对应的元素 得到新的 array
* x, y : 作为输入数组, 需要相同的 shape , 或者可以 broadcastable to same shape.
* 返回 : 与x,y 的broadcast结果相同 shape 的 array



#### 3.4.2.3. 非零选择

* `nonzero(a)`        : Return the indices of the elements that are non-zero.
  * 返回 a tuple of arrays
  * tuple 每个元素是对应 array 元素在该 dim 上的索引
  * 即 len(tuple) == a.ndim
  * `len(tuple[0])` == 非0元素的个数
* `flatnonzero(a)`    : Return indices that are non-zero in the flattened version of a.
  * 最好理解, 将数据扁平化后再查找非0的元素索引, 返回的 array 也是一维的
  * 功能上完全等同于 `np.nonzero(np.ravel(a))[0].`
* `argwhere(a)`       : Find the indices of array elements that are non-zero.
  * `grouped by element.` 每条 index 都作为一行数据返回
  * 功能上几乎等同于 `np.transpose(np.nonzero(a))`  but produces a result of the correct shape for a 0-D array.


## 3.5. Statistics 统计

更加完整的统计函数定义在了这里

### 3.5.1. Averages and variances 平均和方差

### 3.5.2. Histograms


直方图统计, 不止一种

```py
numpy.histogram(a, bins=10, range=None, normed=None, weights=None, density=None)
```

参数:
* a     : array_like , 输入数据, 会被自动 flatten
* bins  : (default=10) int or sequence of scalars or str, 梯度区间.
  * 整数代表固定宽度区间的个数, 具体的区间会根据 range 来计算
  * sequence 代表指定的单调递增区间, 定义了 len(bins)-1个区间, 最后一个区间是闭区间
  * str : 特殊方法的自动计算 edge, 定义在 `histogram_bin_edges`
* range : (float, float), optional
  * 用于定义完整的直方图区间
  * 默认是 (a.min, a.max)
* weights : array_like, optional
  * 和输入数据 a 的shape 一模一样, 用于定义每单个元素的权值
* density : bool, optional
  * 类似于标准化, False 的时候就是普通的统计个数
  * True, 直方图的值会被标准化, sum = 1


返回值 有两个:
* hist  : array
* bin_edges : 因为 bins 可能是整数或者别的省略的输入方法, 该返回值用于标识完整的区间序列
  * 注意 len(bin_edges) = len(hist)+1 

#### 3.5.2.1. bincount 原子统计

直方图的简化版本




## 3.6. Set 集合

### 3.6.1. unique

寻找一组数据中的唯一元素, 可以用来统计元素的种类数  
除了返回独立的元素种类, 还可以返回
* 各个独立元素在原数据中的索引
* (没看懂)the indices of the unique array that reconstruct the input array 
* 各个unique元素在原数据中的出现次数

`numpy.unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None)`
* ar    : 数据, 会自动转换成 1-D, 除非指定 axis, 所以不需要提前 flatten()
* return_index  : bool, 是否返回元数据中索引
* return_inverse: bool, 是否返回索引 (用于 reconstruct ar)
* return_counts : bool, 是否返回统计 (1.9.0 新功能)
* axis          : 用于指定轴




# 4. numpy.random

* numpy 的随机包是独立出来的 `numpy.random`
* 比 Pystl 的 random 包通用性更广, 值得学习

基本概念
1. BitGenerators: 生成随机数的对象, 由随机32 or 64 bit 填入的 unsigned interger 
2. random generator: 主要负责将随机bit转为特定分布
3. Generators: 该库的用户接口, 将 BitGenerator 的随机序列转换为特定的概率分布以及范围的随机数
4. RandomState: 已经被淘汰的旧版生成器, 只能基于一种 BitGenerators
   - RandomState 目前直接作为一个对象被放入了 该命名空间

## 4.1. Generator

基本生成器
* `default_rng(seed=None)`  : 被官方推荐的生成器, 使用 `PCG64` , 效果优于 `MT19937`
  - seed : None, int, array_like`[ints]`, SeedSequence, BitGenerator, Generator. 


### 4.1.1. 生成器方法

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

# 5. numpy 常规功能

## 5.1. numpy 的IO

numpy 的数据IO可以简单分3类:
* 二进制IO
* txt IO
* 与python 内置 string 的转换

numpy 的 IO 也一定程度上基于 pickle, 具有一定的不安全性

通用参数:
* file : file-like object, string, or pathlib.Path
  
### 5.1.1. 类型转换

在 numpy 官方文档中, ndarray 相关的类型转换也被归纳为 IO 的一部分

* ndarray.tolist()
  * 将 np array 转换成python自带的列表, 该函数是拷贝的
  * 会把元素的类型也转成 python 自带的数字类型
  * 对于 0-dim 的 array `np.array(1)`, 直接使用 list(a) 会报异常, 只能用 tolist()
  * 相比与 list(a), a.tolist() 更加安全, 且附带了类型转换
* ndarray.tofile(fid[, sep, format])


### 5.1.2. numpy binary files

最基础的保存方法, 因为是二进制的, 所以最好只通过 numpy 访问, 文件后缀为 `.npy`

* load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
* save((file, arr, allow_pickle=True, fix_imports=True))
* savez
* savez_compressed

### 5.1.3. text file

txt 文件保存后的访问比较便捷, 也容易在其他应用间交互

* numpy.savetxt 将一个 1D or 2D array_like 保存到 txt
  * 注意只能作用在 1/2 维数组
  * 该函数支持非 numpy 的 array (python 自带的数组也可以用)
* numpy.loadtxt(fname, `dtype=<class 'float'>`, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None, *, like=None)

```py
numpy.savetxt(fname, X, 
  fmt='%.18e', delimiter=' ', newline='\n', 
  header='', footer='', comments='# ', 
  encoding=None)
# fmt : str or [str..] : 用于指定数据的记录格式, 小数点等

# delimiter : str, 列之间的分隔符
# newline   : str, 行之间的分隔符

# header    : str, 写在文件最开头的地方
# footer    : str, 写在文件的末尾
# comments  : str, 对于 header 和 footer 的注释符号, 主要用于 numpy.loadtxt 的识别

# encoding : {None, str


numpy.loadtxt(fname, `dtype=<class 'float'>`, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None, *, like=None)
```

# 6. config

## 6.1. np.set_printoptions

1. 取消科学计数法显示数据 `np.set_printoptions(suppress=True)  `



2. 取消省略超长数列的数据 ` np.set_printoptions(threshold=sys.maxsize)` 需要 sys 包


### 6.1.1. numpy.shape

### 6.1.2. numpy.dot()  矩阵点乘

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




## 6.2. linalg 

### 6.2.1. SVD 奇异值分解

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
