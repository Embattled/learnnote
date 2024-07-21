- [1. numpy.array](#1-numpyarray)
  - [1.1. attributes](#11-attributes)
  - [1.2. scalars](#12-scalars)
  - [1.3. 元素运算](#13-元素运算)
    - [1.3.1. universal function](#131-universal-function)
    - [1.3.2. clip 修正边界](#132-clip-修正边界)
  - [1.4. Calculation 降维运算](#14-calculation-降维运算)
  - [1.5. Copies and Views](#15-copies-and-views)
  - [1.6. Indexing routines](#16-indexing-routines)
- [2. NumPy fundamentals](#2-numpy-fundamentals)
  - [2.1. Broadcasting](#21-broadcasting)
    - [2.1.1. General Broadcasting Rules](#211-general-broadcasting-rules)
- [3. Routines array常规操作 API](#3-routines-array常规操作-api)
  - [3.1. Array creation](#31-array-creation)
    - [3.1.1. From shape or value](#311-from-shape-or-value)
    - [3.1.2. From existing data](#312-from-existing-data)
      - [3.1.2.1. From File](#3121-from-file)
      - [3.1.2.2. From Data](#3122-from-data)
      - [3.1.2.3. From Memory](#3123-from-memory)
    - [3.1.3. Numerical ranges 创建范围的array](#313-numerical-ranges-创建范围的array)
  - [3.2. Array manipulation 操纵更改 Array](#32-array-manipulation-操纵更改-array)
    - [3.2.1. Changing array shape 形态转换](#321-changing-array-shape-形态转换)
    - [3.2.2. Transpose-like operations 转置操作](#322-transpose-like-operations-转置操作)
    - [3.2.3. Changing number of dimensions 维度个数操作](#323-changing-number-of-dimensions-维度个数操作)
      - [3.2.3.1. expand\_dims 升维](#3231-expand_dims-升维)
      - [3.2.3.2. squeeze 压缩维度](#3232-squeeze-压缩维度)
    - [3.2.4. Joining arrays 拼接](#324-joining-arrays-拼接)
    - [3.2.5. Splitting arrays 拆分](#325-splitting-arrays-拆分)
    - [3.2.6. Tiling arrays](#326-tiling-arrays)
    - [3.2.7. Adding and removing elements 修改元素](#327-adding-and-removing-elements-修改元素)
      - [3.2.7.1. append](#3271-append)
      - [3.2.7.2. resize 强行更改 shape](#3272-resize-强行更改-shape)
    - [3.2.8. Rearranging elements 重新排列元素](#328-rearranging-elements-重新排列元素)
  - [3.3. Data type routines](#33-data-type-routines)
  - [3.4. Data type information](#34-data-type-information)
    - [3.4.1. Data type testing](#341-data-type-testing)
  - [3.5. Discrete Fourier Transform (numpy.fft)](#35-discrete-fourier-transform-numpyfft)
    - [3.5.1. Standard FFTs 标准傅里叶变换](#351-standard-ffts-标准傅里叶变换)
    - [3.5.2. Real FFTs 复数傅里叶变换](#352-real-ffts-复数傅里叶变换)
    - [3.5.3. Hermitian FFTs](#353-hermitian-ffts)
    - [3.5.4. Helper routines 辅助功能](#354-helper-routines-辅助功能)
  - [3.6. linalg](#36-linalg)
    - [3.6.1. SVD 奇异值分解](#361-svd-奇异值分解)
  - [3.7. numpy Input and Output  Numpy 数据的 IO](#37-numpy-input-and-output--numpy-数据的-io)
    - [3.7.1. NumPy binary files (NPY, NPZ) - 标准Numpy格式的二进制的 io](#371-numpy-binary-files-npy-npz---标准numpy格式的二进制的-io)
    - [3.7.2. Text files](#372-text-files)
    - [3.7.3. Raw binary files](#373-raw-binary-files)
  - [3.8. Linear algebra 线性代数计算](#38-linear-algebra-线性代数计算)
    - [3.8.1. Matrix and vector products 向量矩阵乘法](#381-matrix-and-vector-products-向量矩阵乘法)
      - [3.8.1.1. 矩阵乘法](#3811-矩阵乘法)
      - [3.8.1.2. einsum - Einstein summation convention](#3812-einsum---einstein-summation-convention)
    - [3.8.2. Solving equations and inverting matrices 计算矩阵方程或者逆](#382-solving-equations-and-inverting-matrices-计算矩阵方程或者逆)
  - [3.9. Logic functions 逻辑计算](#39-logic-functions-逻辑计算)
    - [3.9.1. Truth value testing](#391-truth-value-testing)
    - [Comparison - 对比两个 array](#comparison---对比两个-array)
  - [3.10. Masked array operations](#310-masked-array-operations)
  - [3.11. Mathematical function 数学操作](#311-mathematical-function-数学操作)
    - [3.11.1. Trigonometric functions 三角函数](#3111-trigonometric-functions-三角函数)
    - [3.11.2. Hyperbolic functions 双曲线函数](#3112-hyperbolic-functions-双曲线函数)
    - [3.11.3. Rounding 最近值](#3113-rounding-最近值)
    - [3.11.4. Sums, products, differences 求和求积求差](#3114-sums-products-differences-求和求积求差)
    - [3.11.5. Exponents and logarithms 指数](#3115-exponents-and-logarithms-指数)
    - [3.11.6. Rational routines 最大公因数 最小公倍数](#3116-rational-routines-最大公因数-最小公倍数)
    - [3.11.7. Extrema Finding 极值寻找](#3117-extrema-finding-极值寻找)
    - [3.11.8. 杂项](#3118-杂项)
      - [3.11.8.1. convolve 卷积](#31181-convolve-卷积)
      - [3.11.8.2. clip 裁剪](#31182-clip-裁剪)
      - [3.11.8.3. interp 简易线性插值](#31183-interp-简易线性插值)
  - [3.12. Padding Arrays](#312-padding-arrays)
  - [3.13. Polynomials 多项式](#313-polynomials-多项式)
    - [3.13.1. Power Series (numpy.polynomial.polynomial)](#3131-power-series-numpypolynomialpolynomial)
  - [3.14. Random sampling (numpy.random)](#314-random-sampling-numpyrandom)
  - [3.15. Sorting, Searching, Counting 排序 搜索 计数](#315-sorting-searching-counting-排序-搜索-计数)
    - [3.15.1. Sorting 排序](#3151-sorting-排序)
      - [3.15.1.1. 基础排序](#31511-基础排序)
      - [3.15.1.2. 部分有序](#31512-部分有序)
    - [3.15.2. Searching 元素查找](#3152-searching-元素查找)
      - [3.15.2.1. 最大值选择](#31521-最大值选择)
      - [3.15.2.2. 逻辑选择值 where](#31522-逻辑选择值-where)
      - [3.15.2.3. 非零选择](#31523-非零选择)
  - [3.16. Statistics 统计](#316-statistics-统计)
    - [3.16.1. Averages and variances 平均和方差](#3161-averages-and-variances-平均和方差)
    - [3.16.2. Histograms](#3162-histograms)
      - [3.16.2.1. histogram 一维数据直方图](#31621-histogram-一维数据直方图)
      - [3.16.2.2. histogram2d 二维直方图](#31622-histogram2d-二维直方图)
      - [3.16.2.3. bincount 原子统计](#31623-bincount-原子统计)
  - [3.17. Set 集合](#317-set-集合)
    - [3.17.1. unique](#3171-unique)
- [4. numpy.random](#4-numpyrandom)
  - [4.1. Generator](#41-generator)
  - [4.2. Random Generation Function](#42-random-generation-function)
    - [4.2.1. Simple Random 简单的随机生成](#421-simple-random-简单的随机生成)
    - [4.2.2. Permutations 排列](#422-permutations-排列)
    - [4.2.3. Distributions 分布函数](#423-distributions-分布函数)
- [5. Universal functions (ufunc)](#5-universal-functions-ufunc)
- [NumPy's Module Structure](#numpys-module-structure)
  - [numpy.lib](#numpylib)
    - [numpy.lib.stride\_tricks](#numpylibstride_tricks)
- [7. config](#7-config)
  - [7.1. np.set\_printoptions](#71-npset_printoptions)
    - [7.1.1. numpy.shape](#711-numpyshape)
    - [7.1.2. numpy.dot()  矩阵点乘](#712-numpydot--矩阵点乘)

* python数值包 最基础的列表处理包 被其他许多包所依赖  
* python stl 中的 math 有许多同名函数, 但不支持向量输入, 因此机器学习中更多的使用 numpy

# 1. numpy.array

NumPy’s array class is called `ndarray`. It is also known by the alias `array`.  
  
* numpy最基础的类, 所有元素都是同一类型, nd 代表可以是任意维度
* 和 python 自带的 `array.array` 是不同的  
* python 自带的类只能处理一维数组, 并且函数很少

`A = np.array([[1,3,4], [2,3,5], [1,2,3], [5,4,6]])`  
注意 ndarry 的打印特点, 值之间没有逗号, 而是空格

## 1.1. attributes

这些都是属性, 不是方法, 调取不加括号
* ndim      : 维度
* shape     : 元组形式表示各个维度的大小, 等同于 tensor.size()
* size      : 总共的元素个数
* dtype     : 元素类型
* itemsize  : 元素大小 bytes
* data      : 指向实际数据的 `array` 一般不需要使用


* strides: 每个维度访问下一个索引的时候, 内存移动的步幅
  * 是基于 C 的底层 bytes 的步幅, 操纵的时候要非常小心

## 1.2. scalars 

Python 为了便于书写, 对于整数和小数, 各自只定义了一种数据类型: integer float  

numpy 由于需要应用在科学运算上, 数据的类型变得十分重要
* numpy 定义了总计 24 种数据类型  
* 这些数据类型主要基于 C 语言以及 Cpython
* 还有一些数据类型用于保持与 python 内建类型的兼容性



## 1.3. 元素运算

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

### 1.3.1. universal function

* 基础运算之外, 在numpy 包中有一些其他的数学运算, 也是元素层面的
* 也有基础运算的全局函数版
* `np.sin(A) np.exp(A)  np.add(A,B)`

### 1.3.2. clip 修正边界

* numpy.clip(a, a_min, a_max, out=None, **kwargs) 
* 将 array 限制在正确的范围里


## 1.4. Calculation 降维运算

对元素进行某种会带来降维的运算操作, 也可以指定维度 

基本上都有类方法和全局方法两种实现
1. `ndarray.fun(*)`
2. `numpy.fun(array,)`



## 1.5. Copies and Views

* 普通赋值不进行拷贝       `a=b`
* view不进行数据拷贝       `c=a.view()` 算是绑定了一个引用, 可以用来操纵数据
* copy方法会进行完整拷贝   `d=a.copy()`

对于截取数据中的有用片段, 删除无用片段时, 需要进行拷贝
```py
a = np.arange(int(1e8))
b = a[:100].copy()
del a  # the memory of ``a`` can be released.
```


## 1.6. Indexing routines 

有很多包函数没有放在 numpy routines, 而是放在该部分, 以后再读   

# 2. NumPy fundamentals

These documents clarify concepts, design decisions, and technical constraints in NumPy. This is a great place to understand the fundamental NumPy ideas and philosophy.

用于阐述一些非 API 的 numpy 关键性概念


## 2.1. Broadcasting

https://numpy.org/doc/stable/user/basics.broadcasting.html

broadcast 是 numpy 中用于对不同 shape 的 array 进行运算操作的一个名词, 在某些限制下, 较小的数组会 broadcast 到较大的数组的 shape 下, 使得它们之间可以进行运算. 

无需制作不必要的数据 copy, 并且可以帮助算法高效的实现, 然而 某些情况下 自动 broadcast 可能会导致内存使用效率低下导致运算减速  


### 2.1.1. General Broadcasting Rules

普遍 broadcast 规则  
* 两个 array 进行比较时, 从 shape 的最右侧 (即元素层) 的维度开始比较 并向左移动
* 如果两个维度称为兼容的, 则应该满足
  * 这两个维度的长度相同 or
  * 其中一个维度长度为 1
* 否则不可以进行 Broadcast , numpy 会报错  `ValueError: operands could not be broadcast together `


一个常用的例子, 对于 channel-last 的图像进行 scale


```sh
Image  (3d array): 256 x 256 x 3
Scale  (1d array):             3
Result (3d array): 256 x 256 x 3
```

一个反直觉的例子, 两个 array 的 shape 互相交错的等于 1
```sh
A      (4d array):  8 x 1 x 6 x 1
B      (3d array):      7 x 1 x 5
Result (4d array):  8 x 7 x 6 x 5
```



# 3. Routines array常规操作 API

对 Array 数据的各种常规操作

* 尽量明确各个函数的返回值是 copy 的还是只是映射

函数可以分为
1. 类方法和全局函数
2. 返回改变后的值和改变自身(inplace)


## 3.1. Array creation

numpy.array 的各种创建函数能够创建各种各样的预设 array

* 所有函数都有 `dtype` 参数
* 所有函数都有 `order` 参数
* 部分函数里有 `like`  参数 : array_like, optional, 用于将返回值创建成 np.array 以外的数据类型


### 3.1.1. From shape or value

基础创建函数, 需要指定 array 的 shape  
* `*_like` 版本函数, 输入的不再是 shape 而是另一个 array, 相当于 `func(a.shape,...)` 的另一种写法
  * 该版本创建的 array 默认数据类型同原本的 array 一致
  * 具有 _like 版本的函数有 : 

| 函数                                            | 功能                 |
| ----------------------------------------------- | -------------------- |
| `zeros(shape[, dtype, order, like])`            | 全 0                 |
| `ones(shape[, dtype, order, like])`             | 全 1                 |
| `full(shape, fill_value[, dtype, order, like])` | 可以指定初始化的值   |
| `empty(shape[, dtype, order, like])`            | 只分配内存, 不初始化 |

特殊 array
* `identity(n[, dtype, like])`    Return the identity array. 只能是2D正方形, 所以传入的 shape 是单个整数
* `eye(N[, M, k, dtype, order, like])`  创建2d对角线矩阵, 传入的 shape 可以是长方形, k 则代表对角线的偏移


### 3.1.2. From existing data

从既存的数据中创建一个 array, 某种程度上也算是 numpy 的文件 Input

#### 3.1.2.1. From File 

`numpy.fromfile(file, dtype=float, count=- 1, sep='', offset=0, *, like=None)`  文件读取
* `file`  : Open file object or str or Path, 1.17.0 pathlib.Path objects are now accepted.
* `dtype` : 用于指定 array 类型, 同时对于 binary 文件还决定了读取的步长 (即单个 item 的大小)
* `count` : 用于指定要读取的 item 的数量, -1代表读完整个文件
* `sep`   : 指定了该文件是否是 binary 或者 text file, 默认是空字符代表了二进制文件, 如果是空格分隔符 ` `, 则代表 text 文件, 同时分割匹配符会匹配1到多个空白字符
* `offset`: 读取时候的向后偏移, 只在 binary 的时候起作用

#### 3.1.2.2. From Data

`numpy.array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0, like=None)`  
* 从数据中创建一个 np.array , Create an array.
* 这里的 object 可以是 任何具有 array interface 的对象, 或者 nested sequence  
* `copy` : 是否强制拷贝数据, 如果是 False, 则只在各种需要的条件下进行拷贝
* `subok` : 是否传递子类, 如果是 False, 则子类将被强制转换成一个基础类 `base-class`
* `ndmin` : 指定返回值的最小维度, 如果指定的话, 会在传入的 object 的维度不足的情况下在维度`前方`补充
* `like ` : 用于创建非 numpy 支持的类型的 array 

`numpy.asarray(a, dtype=None, order=None, *, like=None)`
* 将已有的数据转换成 array
* 该函数只是一个默认不拷贝的 `numpy.array` 函数

`numpy.asanyarray(a, dtype=None, order=None, *, like=None)`
* 该函数是一个默认 subok=True 的 `numpy.array`

`numpy.ascontiguousarray(a, dtype=None, *, like=None)`
* Return a contiguous array (ndim >= 1) in memory (C order).
* 某种程度上说, 该函数是返回一个所有元素在内存中的位置相邻的 array

`numpy.asmatrix(data, dtype=None)`
* 将数据转化成 matrix, 一种 numpy.ndarray 的子类
* 相当于一个默认不拷贝的 `numpy.matrix` 函数
* 具体的 matrix 数据类型需要另外参照

`numpy.copy(a, order='K', subok=False)`
* 返回一个 array 的主动拷贝
* subok 默认是 False, 即子类会被转化

#### 3.1.2.3. From Memory

`numpy.frombuffer(buffer, dtype=float, count=- 1, offset=0, *, like=None)`
* 从一个 bytes 数据中读取数据
* dtype : 指定数据的类型
* count : 指定按照 dtype 的类型读取的数据个数
* 该函数默认不进行拷贝


`numpy.from_dlpack(x, /)`
* 把一个 DLPACK 数据转换成 ndarray
* 所谓 DLPACK 数据即满足 `__dlpack__ ` protocol 的数据
  * A Python object that implements the __dlpack__ and __dlpack_device__ methods. 



### 3.1.3. Numerical ranges 创建范围的array

该部分的函数都是类似于 range 之类的, 设定开始终止值来生成一个顺序数列

* `numpy.arange([start, ]stop, [step, ]dtype=None, *, like=None)`
  * 完全相当于 python build-in 的函数 range, 但是返回值直接就是 np.ndarray
  * 最好只在参数都是整数的时候使用, 否则用 linspace 替换
* `numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)`
  * 输入起始和终止, 生成均等间隔数列, 支持小数输入 
  * num   : 均分的 `间隔点` 数量, 返回的数列是包括起点在内的间隔点
  * endpoint: bool, 间隔点是否包括终点, 根据值返回数列的间隔会被改变
    * 如果是默认的 True, , 代表了 `[s,e]`闭区间的均等采样 num 个点
    * False, 代表 `[s,e]`闭区间的均等采样 num 个区间, 返回各个区间的起始坐标
  * retstep: bool, 是否返回 step 信息, 如果 True, 则返回的元素是元组 `(samples, step)`

 
* `numpy.meshgrid(*xi, copy=True, sparse=False, indexing='xy')`
  * 从坐标向量生成坐标 mesh, 该函数主要用于作图等其他函数的输入数据, 具体为索引的生成  
  * *xi, 每一个值代表对应维度的最终值, 代表 N-D 的坐标
  * `indexing` : {‘xy’, ‘ij’}, optional. 用于指定坐标的先后顺序
    * xy 代表了图像领域的坐标, x 是横坐标, y 是纵坐标, 但是在数据索引上 y 纵坐标是行数因此靠前, 即坐标顺序是颠倒的
    * ij 代表了普通数列的索引方式, 靠前的维度坐标也靠前
  * 
  * return : X1, X2, ...,XN  根据 len(xi) 和 indexing
    * 代表了该空间下每个点的具体坐标数列  
    * 点的坐标依次是 `( x1[0],x2[0],...,xn[0]  ) , (x1[1],x2[1],..,xn[1])`
    * 每一个返回值的 shape 都是所有输入的 1-D shape 的总和, 可以通过指定 sparse 来实现稀疏的输出

## 3.2. Array manipulation 操纵更改 Array

有各种各样的 array 操作函数, 主要包括 array 形态, 顺序的方面

获取 array 的形态
* `a.shape`
* `np.shape(a)`

拷贝array
* `numpy.copyto(dst, src, casting='same_kind', where=True)`  
  * 把一个 array 的值复制到另一个 array
  * 不同于普通的 copy, 这里 dst 也是提前存在的, 会进行 broadcasting
  * where : array of bool, 附加的元素选择
  * casting : cast模式, 在别的地方应该能学到此处略

### 3.2.1. Changing array shape 形态转换

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


### 3.2.2. Transpose-like operations 转置操作

用于改变维度的`顺序`, 即类似转置的操作

* `numpy.transpose(a, axes=None)`  `a.transpose(*axes)`   `a.T`
  * `axes` : tuple or list of ints 
  * 默认是 `range(a.ndim)[::-1]` 用于完整颠倒 order of the axes
  * 如果指定的话则必须是permutation of `[0,1,..,N-1]` 
  * 返回 A view is returned whenever possible.
  * `a.T` : 懒人函数, 相当于调用默认值的 transpose


* `numpy.moveaxis(a, source, destination)` `numpy.swapaxes(a, axis1, axis2)`
  * transpose 的参数变体, 可能是用于不确定数列 dim 的时候 (transpose 需要输入完整的排列)
  * 只需要确定起始和目标即可, `-1` 经常被用到
  * 返回 view of the array with axes transposed.



### 3.2.3. Changing number of dimensions 维度个数操作

对 array 的维度 (而不是shape)进行操作的函数

atleast_函数集: 注意该输入是 `*arry` 即多参数转化成元组输入, 如果输入多个参数, 则这些参数会被转化成对应维度的 array
* `atleast_1d(*arys)`
* `atleast_2d(*arys)`
* `atleast_3d(*arys)`
* return ndarray

#### 3.2.3.1. expand_dims 升维

朴素升降维:
* `numpy.expand_dims(a, axis)`
  * axis : int or tuple of ints. 范围是 (0,a.ndim)
  * 被插入的维度的位置 == axis , 意思是其后的维度会被顺延
  * 被插入的维度的 shape 都是 1
  * return : `View` of a with the number of dimensions increased. 因为不需要更改内存分配

1. `arr=arr[:,:,np.newaxis`
2. `arr=np.array([arr])`


#### 3.2.3.2. squeeze 压缩维度 
* `numpy.squeeze(a, axis=None)`
  * 删掉 shape 为 1 的维度
  * axis : None or int or tuple of ints, optional. 可以指定, 但是指定的维度必须确保 shape == 1
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

`numpy.split(ary, indices_or_sections, axis=0)`
* 把一个 array 拆成复数个 sub-array, 返回值是一个 list
* indices_or_sections: 
  * int, 如果是单个整数, 则会对指定的 axis 进行均分, 不能均分就报错
  * 1-D sorted array, 会作为切分点, 生成 len(indices)+1个子数列, 如果 index 超过了shape, 则会生成空数列, 不会报错
* axis : 默认值是 0 

`numpy.array_split(ary, indices_or_sections, axis=0)`
* 上面函数的允许不整除版本
* 如果 indices_or_sections 是整数且不能整除, 则
  * 返回 shape % i 个长度为 shape//i + 1 的数列, 剩下的长度为 shape // i

三个懒人函数:
* 相当于指定 axis=0 `numpy.vsplit(ary, indices_or_sections)`  
* 相当于指定 axis=1, 对于 1维数列不报错并执行 axis=0,`numpy.hsplit(ary, indices_or_sections)`  
* 相当于执行 axis=2 `numpy.dsplit(ary, indices_or_sections)`

### 3.2.6. Tiling arrays

平铺一个 array, 沿着某一个维度复制整个 array

* `numpy.tile(A, reps)` : Construct an array by repeating A the number of times given by reps.
  * reps : array_like. The number of repetitions of A along each axis.
  * 输入的 reps 即重复次数, 但是 reps 自身也可以是一个 array 
  * 输出的结果的 ndim 会是 max (len(reps), A.dim)
  * 如果 A.dim < len(reps), 那么 A 将会被前补 dim, 即 (2,2) -> (1,2,2). 如果这不是想要的形式则需要手动 expand_dim 在使用该函数
  * 如果 A.dim > len(reps), 那么 reps 将会被前补 dim, 即 (2,2) -> (1,2,2). 如果这不是想要的形式则需要手动 expand_dim 在使用该函数

* `numpy.repeat(a, repeats, axis=None)` : Repeat elements of an array.
  * repeats : int or array of ints. 
    * The number of repetitions for each element. 
    * 指定对于每一个 index 上的元素要重复几次
  * axis : int, optional. 该函数默认会使用并输出 flatten 的 array, 因此大多数时候需要指定 axis


### 3.2.7. Adding and removing elements 修改元素

将 np.ndarray 以类似于普通 list 的视角操作

#### 3.2.7.1. append

该函数不存在 in-place 模式, Append values to the end of an array.

* `numpy.append(arr, values, axis=None)` 
  * arr : Values are appended to a `copy` of this array.
  * values: array_like. 维度 ndim 必须和 arr 一样
  * axis : None 要被插入的维度
    * values : 必须是 correct shape (the same shape as arr, excluding axis).
    * if axis = None, values can be any shape and will be flattened before use
    * If axis is None, out is a flattened array.
  * return : A copy of arr with values appended to axis. 
  

#### 3.2.7.2. resize 强行更改 shape

不同于 reshape 的resize
* 会带有填充以及裁剪的更改 array 形态
* 以 C-order进行
* `numpy.resize(a, new_shape)` : 不足的地方用 a 的重复来填充, 返回新的 array
* `ndarray.resize(new_shape, refcheck=True)` in-place 的修改, 不足的地方会用 0 填充
  * 如果 refcheck 是True, 那么如果该 array 有别的引用的话会发生 raise 来停止该操作
* 


### 3.2.8. Rearranging elements 重新排列元素  

最经典的 reshape 也被包括在这里, 但是上面写了这里就省略

flip 以及 懒人 flip : Reverse the order of elements in an array along the given axis
* `numpy.flip(m, axis=None)`  
  * axis : None or int or tuple of ints, optional. 如果 axis 是默认值 None, 那么所有维度都会一起反转
* `numpy.flipud(m)`
  * 反转上下, 即行, axis = 0
* `numpy.fliplr(m)`
  * 反转左右, 列, 要求输入数据必须是 2维以上, axis=1

## 3.3. Data type routines 

在需要谨慎操作不同数据之间的类型的时候, 使用该部分接口来进行验证  

## 3.4. Data type information

* `class numpy.iinfo(type)`
  * 从一个  integer type, dtype, or instance 中创建一个 整数的 信息类
  * min, max : 获取该整数类型的最小最大值
  * dtype: 返回具体的 dtype 
  * bits : 返回该类型所占据的 bits 数

* `class numpy.finfo(dtype)`
  * 类似的 从 float, dtype, or instance 中创建一个 浮点数的信息类
  * 官方提示不要在模组层定义该类的实例, 因为计算量很重
  * 可访问的属性超级多...

### 3.4.1. Data type testing


* `numpy.issctype(rep)`
  * 验证一个输入是不是 scalar data-type , 返回 bool
  * 要注意字符串也是 scalar, `np.int32` 等
  * 不是 scalar 的可以是 `list` `1.1`小数等


* `numpy.issubdtype(arg1, arg2)`
  * 有点类似于 python 的内置函数 `issubclass`, 但是是针对 numpy dtype 的
  * 返回 True 如果 arg1 是 typecode <= arg2 的
  * 不能用来进行位宽 size 的比较 `np.float64 np.float32` 之间总是 false
  * `np.issubdtype(floats.dtype, np.floating)` True
  * `np.issubdtype(ints.dtype, np.integer) ` True


* `numpy.issubsctype(arg1, arg2)`
  * 更加抽象了, 似乎和 numpy 本身没什么关联, 说明也是  if the first argument is a subclass of the second argument.
  * `np.issubsctype('S8', str)` False
  * `np.issubsctype(np.array([1]), int)` True
  * `np.issubsctype(np.array([1]), float)` False

## 3.5. Discrete Fourier Transform (numpy.fft)

The SciPy module `scipy.fft` is a more comprehensive superset of numpy.fft, which includes only a basic set of routines.
* numpy 的傅里叶变换包只提供了一些基础的 routines 
* 在满足使用场景的情况下, 使用 numpy 的包可以不用导入 scipy

numpy fft 的细则:
* 类型提升
  * 为了保证精度, numpy.fft 会自动对输入数据进行数据精度提升, `float32 -> float64` 以及 `complex64 -> complex128`
  * 对于不提升精度的 FFT 实现, 需要参考 `scipy.fftpack`
* Normalization : 傅里叶变化的标准化选项
  * 所有的 numpy.fft 实现都有 `norm` 选项代表 标准化 选项, 有三种
  * `backward`
  * `ortho`
  * `forward`
* 


### 3.5.1. Standard FFTs 标准傅里叶变换

### 3.5.2. Real FFTs 复数傅里叶变换

### 3.5.3. Hermitian FFTs 


### 3.5.4. Helper routines 辅助功能

* `np.fft.fftshift(x, axes=None)`: 便于 fft 结果的可视化
  * 标准 np.fft 运算的结果, 0频率的部分位于结果数列 `x[0]`, 然而一般为了可视化, 更加倾向于将 0 频率的部分移到数组中间
  * `axes=None`,  int or shape tuple, optional, 很重要, 需要同 fft 应用时候的 axis 参数一致
  * 注意, 最终 shift 的结果上, `y[0]` 是 Nyquist component only if `len(x)` is even.

* `np.fft.ifftshift(x, axes=None)` : fftshift 的逆运算
  * identical for even-length x
  * differ by one sample for odd-length x


## 3.6. linalg 

### 3.6.1. SVD 奇异值分解

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

## 3.7. numpy Input and Output  Numpy 数据的 IO

numpy 对于各种类型的输出支持的很好, 要注意对于 pandas 的 DataFrame 支持写在了别的段里  

通用参数: 
* fmt : str or sequence of strs, optional
  * 用于指定数据在输出时候的格式
  * 可能在别的地方有完整的文档, 保留为 [TODO]

### 3.7.1. NumPy binary files (NPY, NPZ) - 标准Numpy格式的二进制的 io

最基础的保存方法, 因为是二进制的, 所以最好只通过 numpy 访问, 文件后缀为 `.npy` or `.npz`

* load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
  * 对于 npy 文件,  a single array is returned.
  * 对于 npz 文件 dictionary-like object is returned, containing {filename: array} key-value pairs, one for each file in the archive.

* save((file, arr, allow_pickle=True, fix_imports=True))
* savez   : 特殊格式 .npz 的存储
* savez_compressed  : 带压缩的 .npz 的存储



* `numpy.load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII', *, max_header_size=10000)`
  * 从 numpy 标准二进制格式文件 `.npy .npz` 中读取一个 arrays or objects
  * `file` : 支持  file-like object, string, or `pathlib.Path` 
  * `mmap_mode = None`  : 文件读取时候的内存映射.
    * 主要用于 memory-mapped array, 由于其数据是保存在 disk 上的, 但是仍然可以通过正常的切片进行访问.
    * mmap_mode 主要参考 `numpy.memap` 
    * 对于访问大文件的小片段而不将整个文件读取内存的应用场景特别有用
  * `allow_pickle=False` : 是否允许加载使用 pickle 打包的数据对象, 由于 pickle 的安全问题默认为 False
  * `fix_imports=True` : 是否兼容 python2 的 pickled 文件, 默认为真
  * `encoding='ASCII'` : 也是用于兼容 python2 的 pickled 文件的, 支持  ‘latin1’, ‘ASCII’, and ‘bytes’ 
  * `max_header_size=10000` : 需要参照 python 语言的语法类 `ast.literal_eval`, 过大的 headers 可能会不安全.  

* `numpy.save(file, arr, allow_pickle=True, fix_imports=True)`
  * `file`, file, str, or pathlib.Path. 注意, 这个保存路径可以不包括 `.npy` 的后缀, 如果没有该后缀的话会被自动加上
  *  arr :要保存的 array

* `numpy.savez(file, *args, **kwds)`  : 将多个array作为 dict 存入 .npz 文件中
  * 如果要手动指定 array 在访问时候的名称, 则使用关键字参数传入 `savez(fn, x=x, y=y)`
  * 如果作为位置参数传入 array, 则会自动命名为 `arr_0, arr_1, etc.`


### 3.7.2. Text files

以可以直接读取的 txt 文件来存储数据  
txt 文件保存后的访问比较便捷, 也容易在其他应用间交互

保存 save :
* `numpy.savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)`
  * 将一个 1D or 2D array_like 保存到 txt, 该函数支持非 numpy 的 array (python 自带的数组也可以用)
  * X : array , 注意只能作用在 1/2 维数组, 只支持 1D 或者 2D
  * delimiter : 2D 数据的时候分割列的符号
  * newline : 1D 数据或者 2D数据分割行 的符号
  * header  : 很方便, 在参数里就能直接传入数据的首列标识符, 还会贴心的加上 `#`
  * footer  : 同理, 添加到文件末尾  
  * comments: 会被添加到 header 和 footer 

* `numpy.loadtxt(fname, `dtype=<class 'float'>`, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None, *, like=None)`

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
```


### 3.7.3. Raw binary files

操作系统以及 numpy 库无关的纯二进制式文件存取, 可以被用于 Raw 图像的IO


* `numpy.fromfile(file, dtype=float, count=-1, sep='', offset=0, *, like=None)`
  * 从一个 binary or txt file 中构建一个 array, 即本身也支持从 txt 来读取
  * 更多的时候是与 `tofile` 接口配合使用
  * 在知道数据类型 data-type 的时候, 是一个更加高效的读取数据的方法.
  * `file`: file or str or Path, Open file object or filename.
  * `dtype=float` : binary 数据的格式, 主要用于决定 byte-order 以及具体的数据大小, 支持大部分 builtin types
  * `count=-1` : Number of items to read. -1 means all items. 这里的 items 的意思不太懂, 应该是 array 的单个元素
  * `sep=''` : 用于 txt file 的读取, 当默认值的时候, 即 empty separator 的时候代表该文件是 binary.
  * `offset=0` : 同理, 基本的文件读取的接口, 用于 offset (in bytes) from the file’s current position. 只在 binary 的时候有效
  * `like=None` : 与 `like` 关键字有关, Reference object to allow the creation of arrays which are not NumPy arrays


* `ndarray.tofile(fid, sep='', format='%s')`
  * 将一个 array 作为 binary(默认情况) 或者 txt 写入文件
  * `fid` : file or str or Path, An open file object, or a string containing a filename.
  * `sep=''` : 默认情况下作为 binary 写入 等同于 `file.write(a.tobytes())`, 当 sep 不为空的时候作为 txt. 
  * `format='%s'` : 当文件作为 txt 输出的时候, 用于对每个元素进行格式化输出, 具体表现为输出的内容为 ` "format" % item`

## 3.8. Linear algebra 线性代数计算 

包含了 numpy 的各种线性代数计算函数, 其中一些函数定义在子包 `numpy.linalg` 中  
* numpy 的线性代数函数基于各种 BLAS 和 LAPACK 库来实现, 提供了高速的各种线性代数计算
* 对于性能要求严格的使用场景, numpy 官方更加推荐使用环境依存的线性代数库, e.g. OpenBLAS, MKL, ATLAS, 这些库提供了对多线程和多核心的完整支持
* 对于 `SciPy`库, 也提供了对应的 线性代数部分 `scipy.linalg`, 和 numpy 有一定的重合
  * pros scipy 的库更加全面, 有一些 numpy 中没有的计算函数, 对于重合的函数, scipy 也有一些额外的参数
  * cons numpy 的有些函数对于 array 的 broadcasting 的效果更好

### 3.8.1. Matrix and vector products 向量矩阵乘法

向量和矩阵的乘法  


#### 3.8.1.1. 矩阵乘法

* `numpy.dot(a, b, out=None)`  矩阵点乘, 可以理解为尽可能执行矩阵乘法, 对于高维是有一定拓展性的, 但是不适用于 Tensor
  * 因为是函数, 所以不存在手动加 T 之类的, 只根据 a,b 的维度来决定操作
  * a,b 都是 1D, 执行 向量内积
  * a,b 都是 2D, 执行矩阵乘法, 即 `numpy.matmul` 或者 `@` 运算符
  * a,b 有一个标量, 执行普通元素乘法
  * a,b 有一个 ND, 和一个1D, 对于 a,b 的最后一维执行 sum product (对应元素求积再相加), 矩阵向量乘
  * a,b 是 ND和MD, 执行 sum product over the last axis of a and the second-to-last axis of b. 
    * 最终得到结果为 (N+M-2) 维, 即其余维进行全排列, 这对于某些场景来说可能不适用
    * `dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])`

* `numpy.linalg.multi_dot(arrays, *, out=None)`  多个矩阵相乘
  * `arrays` : 矩阵的 array, 里面的矩阵按顺序相乘, 矩阵都是标准2D矩阵
  * 众所周知矩阵连续乘法结合律, 顺序不同会影响总体的计算量
  * 该函数会自动选择最快的结合方式来实现 arrays 中的矩阵相乘

* `numpy.tensordot(a, b, axes=2)` tensor 矩阵乘, 仅仅只是指定维度的相乘并相加
  * a, b 要执行点乘的 tensor
  * `axes` : int or (2,) array_like , axes to be summed over
    * 如果是 int 则选择 a 的最后 N 维和 b 最前 N 维
    * 被 axes 指定的 a,b 进行积和的结果是单个标量
  * a,b 没有被选中的 axes 也是进行全排列, 即结果的维度是 (a.ndim + b.ndim - len(axes.a) - len(axes.b))

* `numpy.matmul(x1, x2, /, out=None, *, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj, axes, axis]) = <ufunc 'matmul'>`
  * 完备的矩阵乘法, 支持矩阵 stack, 即多维
  * 使用 x1,x2 的 `最后两维` 作为矩阵进行乘法, 其他维度则进行 broadcast 
  * x1,x2 如果是 1D, 仍然会被正确的视作 行/列

```py
a = np.ones([9, 5, 7, 4])
c = np.ones([9, 5, 4, 3])

np.dot(a, c).shape :(9, 5, 7, 9, 5, 3)
np.matmul(a, c).shape :(9, 5, 7, 3)
```

* `numpy.outer(a, b, out=None)` 向量外积, 非拓展
  * 该函数比较基础, 只接受 a,b 都是向量
  * 生成外积矩阵, 矩阵形状为 (a.len, b.len)

#### 3.8.1.2. einsum - Einstein summation convention

评估操作数的爱因斯坦求和约定
Evaluates the Einstein summation convention on the operands.

爱因斯坦求和约定, 可以用一个简单的方式表示许多常见的 多维线性数据运算.  
在 numpy 中
* 隐式implicit模式下, 直接计算对应的值
* 显式explicit模式下, provides further flexibility to compute other array operations that might not be considered classical Einstein summation operations.
  * 通过禁用或者强制 对指定下标的标签进行求和, 进一步提高了运算的灵活性

`numpy.einsum(subscripts, *operands, out=None, dtype=None, order='K', casting='safe', optimize=False)`
* `subscripts` : str, 用字符串来表示求和的格式.
  * 包含显示指示符 `->` 以及精确输出形式的下标标签的情况下执行显示运算
  * 否则执行隐式计算
* `operands` : list of array_like, 输入的操作数

subscripts 是一个以逗号分割的下标标签列表, 每个标签指的是响应操作数的一个维度, 每当一个标签重复时 代表指定维度的值会被相乘并求和
* ` ("i,i",a,b) `  = np.inner(a,b)
* ` ("i", a) `  = a.view()
* ` ("ij,jk", a,b) ` = np.matmul(a,b) 相当于传统矩阵乘法
* ` ("ii" , a) ` = np.trace(a) 同一个操作数取相同的标签, 代表提取对角线 

在隐式模式下, 输出的轴的顺序会按照字母顺序重新排序, 因此标签的选取很重要
* ` ("ij", a) ` 不会对 2D 数组进行更改
* ` ("ji", a) ` 对 2D 数组进行转置
* ` ("ij,jh, a,b) ` 返回矩阵乘法的转置

显示模式下, `->` 直接控制输出下标标签列表, 可以实现禁用或者强制求和
* ` ("i->", a) ` = np.sum(a, axis=-1)  
* ` ("ii->i", a) ` = np.diag(a) 
* ` ("ij,jh->ih", a,b) ` 同显式模式不同, 可以正确获取矩阵乘法
* 显式模式下对操作符默认不进行 broadcasting

当 operand 只有一个, 且不进行任何求和类型的操作的时候, 返回的是 view 

### 3.8.2. Solving equations and inverting matrices 计算矩阵方程或者逆

* `numpy.linalg.inv(a)`  计算一个矩阵的逆
  * 具体在代码上表现为 `dot(a, ainv) = dot(ainv, a) = eye(a.shape[0])`  
  * 注意, 这个 a 是支持多维的, 只需要满足 `(...,M,M)` 即可, 自动使用后两维作为矩阵
  * 返回值 ainv 也满足 shape `(...,M,M)`


## 3.9. Logic functions 逻辑计算

包括 ndarray 之间的逻辑运算以及自身元素的检查逻辑

### 3.9.1. Truth value testing

只有两个函数
* `numpy.all(a, axis=None, out=None, keepdims=<no value>, *, where=<no value>)` 是否全部为 True
* `numpy.any(a, axis=None, out=None, keepdims=<no value>, *, where=<no value>)` 是否有 True


### Comparison - 对比两个 array

这个分组的函数很有意思

* `numpy.array_equiv(a1, a2)` : 比较两个array是否相同, 允许 broadcast
* `numpy.array_equal(a1, a2, equal_nan=False)` : 比较值和 shape 是否都相同, 即不允许 broadcast
  * equal_nan 主要用于负数, 如果实部或者虚部为 nan, 则直接判断该元素相同

## 3.10. Masked array operations 

同 Logic functions 操作非常相似, 主要是通过各种逻辑判断来生成 mask 数据  



## 3.11. Mathematical function 数学操作

绝大多数常用的数学基础函数都属于该分类
* Trigonometric functions 三角函数
* Hyperbolic functions 双曲线函数
* Sums, products, differences 求和, 求积

### 3.11.1. Trigonometric functions 三角函数


基础三角函数 : 输入 radians 的角度
* sin
* cos
* tan
* arcsin
* arccos
* arctan


角度数和弧度的转换:
* radians, deg2rad  : degrees to radians   `deg2rad(x)=  x*pi / 180`
* degrees, rad2deg  : radians to degrees   `rad2deg(x)=  180 * x / pi`


### 3.11.2. Hyperbolic functions 双曲线函数



### 3.11.3. Rounding 最近值

### 3.11.4. Sums, products, differences 求和求积求差

`a, axis=None, dtype=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>` : 
* numpy.prod      : 求所有元素的乘积
* numpy.sum       : 求所有元素的加法和
* numpy.nansum    : 带有 Nan 考虑的加法和, 将 Nan视为 0 
* numpy.nanprod   : Nan视为 1

`a, axis=None, dtype=None, out=None`  : 累计求解 返回的 array 和输入是相同的, axis 只能指定整数
* numpy.cumprod   : 累乘
* numpy.cumsum    : 累加
* numpy.nancumsum : 带有 Nan 考虑的加法和, 将 Nan视为 0, 结果中 Nan 会被替换为前导的累计和
* numpy.nancumprod   : Nan视为 1, 同理


`numpy.diff(a, n=1, axis=-1, prepend=<no value>, append=<no value>)`
* 沿着对应 axis (默认是最后一维) 计算元素差, 即 `out[i]=a[i+1]-a[i]`
* diff 的 `n` 就是在结果上重复做 n 次 diff, 最终的输出会在 输入的基础上, axis 维度的 shape 减少 n

`numpy.ediff1d(ary, to_end=None, to_begin=None)`
* 本身计算上同 diff 相同, 但是只能是 `n=1`
* 提供了两个 to_end to_begin 两个参数, 在结果的基础上附加上参数的 array


`numpy.gradient(f, *varargs, axis=None, edge_order=1)`
* 计算梯度, 输出的 shape 和输入相同, 有点复杂


`numpy.cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None)`
* 计算两个向量的 叉乘, 当 a,b 都是 R3 的时候, 叉乘的结果向量与 a,b 都垂直
* 如果 a,b 都是向量的 array, 则默认情况下 最后一维代表了向量本身
* 返回值称为 c
* axis(a,b,c), axis 是统一的设定, 会覆盖对 a,b,c 的单独参数
* 如果对应向量的维度的某一个只有 2, 则会对输入填充 0
* 如果两个输入的维度都只有 2, 则会返回  the z-component of the cross product is returned.

利用 cross 和 eye 可以不利用 Scipy 就可计算矩阵的反对称矩阵  
有 `skew_symmetric = np.cross(np.eye(3), a)`


### 3.11.5. Exponents and logarithms 指数
### 3.11.6. Rational routines 最大公因数 最小公倍数

### 3.11.7. Extrema Finding 极值寻找

应该是用的比较多的一类方法, 总的上来说就 min 和 max, 但是根据使用场景的不同分出了四种

* `axis=None` None or int or tuple of ints, optional
   通过指定函数的 `axis=i` 参数, 可以指定运算所执行的对象
    * `i` 从0开始, 指代最外层的括号, 即索引访问时最左边的方括号
    * 其他维度的索引每取一个值, 对 所有 i 维度的值进行运算



* `numpy.maximum`  `numpy.minimum`   : 元素维度的比较
  * `(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])`
  * 比较 x1 和 x2 两个数列, 这两个数列的 shape 要一直或者能被 broadcast
  * 返回值理论上和 x1, x2 shape相同
  * 如果某个元素是 NaN, 则返回Nan, 无论是 maximum 还是 minimum
  * 对于复数来说, 只要实部或者虚部有 NaN 那么这个数就被认作是 NaN
  * 该函数会尽可能保留 NaN

* `numpy.fmax` `numpy.fmin`
  * 与 maximum 和 minimum 唯一的区别是在遇到 NaN 的时候, 会保留非 NaN 的一方
  * 还函数会尽可能忽视 NaN

* `amax` `amin`
  * `(a, axis=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)`
  * axis : 支持 tuple 输入, 但是不支持 list
  * keepdims : 对于 axis 指定的维度会被 squeeze, 输入该值为 True 来使得对应的维度保留 shape = 1
  * 该函数会传播 NaN, 只要比较的元素中有 NaN, 则返回 NaN

* `nanmax` `nanmin`
  * `(a, axis=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)`
  * 同 amax 完全一致, 但是对于 NaN 会无视
  * 如果某一个比较的 slices 都是 NaN , 则会报警告, 并返回 NaN

### 3.11.8. 杂项

有时候需要的很特殊的功能, 没办法分类, 目前学习的有

* convolve
* clip
* interp

#### 3.11.8.1. convolve 卷积

Returns the discrete, linear convolution of two one-dimensional sequences.
* 常用在信号处理中
* 一维卷积
* 可以用来非常快的计算一维数据的移动平均 
  * `np.convolve(data, np.ones(平均区间长)/平均区间长)`

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

#### 3.11.8.2. clip 裁剪

`numpy.clip(a, a_min, a_max, out=None, **kwargs)`
* 裁剪一个 array, 比用最大最小值实现要快, 且代码更清晰
* Equivalent to but faster than `np.minimum(a_max, np.maximum(a, a_min))`
* a_min, a_max : 可以指定成 None, 表示为没有任何检查, 只能有一个被指定成 None
  * 注意 a_min a_max 是没有大小比较验证的, 需要用户自己保证


#### 3.11.8.3. interp 简易线性插值

`numpy.interp(x, xp, fp, left=None, right=None, period=None)`  
One-dimensional linear interpolation for monotonically increasing sample points.
* 限制比较多, 只支持 1D, 且数列需要单调递增, 文档建议使用 scipy 的相关插值函数
* x ： 要进行插值的坐标
* fp : 被插值的数列,
* xp : fp 各个元素在 x 轴上的具体坐标, 必须是单调递增的, 除非指定 period, 不能包含 NaN.
* left, right. 用于指定 x 在超越界限 (大于 `x[-1]` 小于 `x[0]` ) 的时候的输出值
* period : A period for the x-coordinates, xp 的周期, 一般用来计算角度, 即360 度为一圈, 720度会被正确的放在 0 度的位置


## 3.12. Padding Arrays

numpy 的填充函数, 只有一个函数单独作为了一类, 可以对任意维度进行填充

`numpy.pad(array, pad_width, mode='constant', **kwargs)`  
* array : array, array_like of rank N
* pad_width: {sequence, array_like, int}
  * `((before_1, after_1), ... (before_N, after_N))` unique pad widths for each axis. 指定每一个维度的始末填充的宽度
  * `(before, after)` or `((before, after),)` 为所有维度指定相同的始末填充宽度
  * `(pad,)` or `int` 等同于为所有维度指定 始末相同的填充宽度 
* mode : str, 用于表示填充操作的种类
  * 


## 3.13. Polynomials 多项式

numpy 1.4 引进的多项式包, 是对于之前的函数包 `numpy.poly1d` 的扩展  

概念区分:
* `polynomial module` 指的是 old API, 定义在了 `numpy.lib.polynomial` 里, 其中包括
  * 旧的多项式类 `numpy.poly1d`
  * 其他的 polynomial functions
* `polynomial package` 指的是 new API, 定义在了 `numpy.polynomial`
  * convenience classes for the different kinds of polynomials
  * 官方推荐在书写新代码的适合使用 numpy.polynomial

该部分决定只阅读新 API numpy.polynomial:
* 支持多种多项式类型, 包括 : Chebyshev, Hermite (two subtypes), Laguerre, and Legendre polynomials
* 每种多项式类型都提供一个独特同时接口统一的类来操作


具体包括: 6 种多项式的子包和对应的class, 和一个统一的工具
* Power Series (numpy.polynomial.polynomial)          最基础的多项式, 其他的都是新加入的  
* Chebyshev Series (numpy.polynomial.chebyshev)
* Hermite Series, “Physicists” (numpy.polynomial.hermite)
* HermiteE Series, “Probabilists” (numpy.polynomial.hermite_e)
* Laguerre Series (numpy.polynomial.laguerre)
* Legendre Series (numpy.polynomial.legendre)
* Polyutils 需要手动精确导入
  

6 种多项式的 class 是不需要进入到子模组进行使用的, `np.polynomial` 的 `__init__` 中已经将6种类导入了
```py
from .polynomial import Polynomial
from .chebyshev import Chebyshev
from .legendre import Legendre
from .hermite import Hermite
from .hermite_e import HermiteE
from .laguerre import Laguerre
```


### 3.13.1. Power Series (numpy.polynomial.polynomial)

提供了一些用于处理多项式的接口, 包括一个 `Polynomial` 类 以及其他的方便接口

基本上对于多项式的处理都是面向对象的操作形式, 类提供了标准数值运算接口   
`‘+’, ‘-’, ‘*’, ‘//’, ‘%’, ‘divmod’, ‘**’, and ‘()’`  

类 : `class numpy.polynomial.polynomial.Polynomial(coef, domain=None, window=None, symbol='x')`
* `coef` : array_like, 用以表示系数, 以 degree 增加的顺序来表示
  * (1,2,3) 会生成 `1+2*x+3*x**2`
* 区间映射:
  * `domain` : (2,) array_like, optional, The default value is `[-1, 1]`.
  * `window` : (2,) array_like, optional, The default value is `[-1, 1]`.
  * 区间缩放以及映射, 会根据将 `(domain[0],domain[1])` 的数值映射到 `(window[0],window[1])` 的缩放系数来决定最终输出数据的缩放和偏移
  * `symbol` : (New in version 1.24.)str, optional. 在打印多项式的方程式的时候, 用于表示变量的字符, 默认是 `x`. 该接口有点新, 不建议使用.


Methods:
* `__call__(arg)`  : 将多项式作为一个函数来调用
* copy()          : 返回一个多项式的拷贝



## 3.14. Random sampling (numpy.random)

## 3.15. Sorting, Searching, Counting 排序 搜索 计数

这里的 counting 都是很简单的函数, 更详细的统计在 statistics 模块


### 3.15.1. Sorting 排序

#### 3.15.1.1. 基础排序

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


#### 3.15.1.2. 部分有序

* partition(a, kth[, axis, kind, order])    :  Return a partitioned copy of an array.
* argpartition(a, kth[, axis, kind, order]) : 

### 3.15.2. Searching 元素查找

大概可以分成
* 极值查找
  * argmax
  * argmin

* 逻辑查找
  * where
* 非零查找
  * argwhere
  * nonzero
  * flatnonzero

#### 3.15.2.1. 最大值选择

* `argmax(a[, axis, out, keepdims])`
  * Returns the indices of the maximum values along an axis.
  * 一般用于机器学习通过可信度得到最终 label
* `argmin(a[, axis, out, keepdims])`
  * Returns the indices of the minimum values along an axis.



#### 3.15.2.2. 逻辑选择值 where

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



#### 3.15.2.3. 非零选择

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


## 3.16. Statistics 统计

更加完整的统计函数定义在了这里

### 3.16.1. Averages and variances 平均和方差

较为通用的统计函数, 根据对于 NaN 的处理分为标准版和 `nan*` 版  (average 除外没有 nan 版本) 

通用参数:
* `axis` : 输入 a 会被默认扁平化, 除非指定 axis
* `keepdim`: 对于被 reduced 的维度是否保留 (保留的化 shape 为 1)
* `dtype`: 默认的数据类型(float64 for int input, same with input for float input)
* `where` : array_like of bool, optional 指定各个元素是否参与运算的布尔 array


基础统计函数
* `numpy.average(a, axis=None, weights=None, returned=False, *, keepdims=<no value>)`
  * 拥有 weights 权重, 即可以权重平均

* `numpy.median(a, axis=None, out=None, overwrite_input=False, keepdims=False)`
  * 中值

* `numpy.mean(a, axis=None, dtype=None, out=None, keepdims=<no value>, *, where=<no value>)`  
  * 算数平均, 即没有权重的平均
* `numpy.var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=<no value>, *, where=<no value>)`
  * `ddof` int, optional. Delta Degrees of Freedom, the divisor used in the calculation is `N - ddof`. 自由度

* `numpy.std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=<no value>, *, where=<no value>)`
  * 标准差


### 3.16.2. Histograms


直方图统计, 在 Statistic 分类的函数中属于一个大类, 不止一个函数  

* numpy.histogram    : 普通一维直方图统计, 


#### 3.16.2.1. histogram 一维数据直方图
```py
numpy.histogram(a, bins=10, range=None, normed=None, weights=None, density=None)
```

参数:
* a     : array_like , 输入数据, 会被自动 flatten
* bins  : (default=10) int or sequence of scalars or str, 梯度区间.
  * 整数代表固定宽度区间的个数, 具体的区间会根据 range 来计算
  * sequence 代表指定的单调递增区间, 定义了 `len(bins)-1`个区间
  * str : 特殊方法的自动计算 edge, 定义在 `histogram_bin_edges`
* range : (float, float), optional
  * 用于定义完整的直方图区间
  * 默认是 (a.min, a.max)
  * 超出范围的数据将被视为 outlier 不纳入直方图统计中
* weights : array_like, optional
  * 和输入数据 a 的shape 一模一样, 用于定义每单个元素的权值
* density : bool, optional, default=None
  * 类似于标准化, False 的时候就是普通的统计每个 bin 里的数据个数
  * True, 直方图的值会被标准化, 在均等的 bin 的情况下 sum = 1, 但是非均等的 bin 下好像 sum != 1, 原因没看懂
  * ` it is not a probability mass function.`
* normed : default None (Deprecated)
* return : 
  * hist : array
  * bin_edges : array of dtype float,  Return the bin edges `(length(hist)+1)`. 返回所有区间的 edges, 包括前后




返回值 有两个:
* hist  : array
* bin_edges : 因为 bins 可能是整数或者别的省略的输入方法, 该返回值用于标识完整的区间序列
  * 注意 len(bin_edges) = len(hist)+1 

#### 3.16.2.2. histogram2d 二维直方图

Compute the bi-dimensional histogram of two data samples.   
并不是单纯的二维数据直方图统计, 而是一种双方向上的统计.  
输入数据并不是 (n,m) 的二维数据, 而是 (n,2) 的在二维平面上的点集, 然后根据 bins 将数据范围平面分成一个个矩形, 然后统计每个矩形中的点的个数.  
输入参数的 x,y 相当于把所有点的坐标分开输入  


```py
numpy.histogram2d(x, y, bins=10, range=None, normed=None, weights=None, density=None)
```
参数说明:
* x,y  : array_like, shape (N,)  代表了 x, y 坐标轴上的点. 两个数据维度应该相同
* bins : `int` or `array_like` or `[int, int]` or `[array, array]`
  * 整数的时候代表 bins 的个数
  * 如果是 array, 则代表 bins 的 edges, 定义同一维的 histogram 相同
  * 也可以 `[int,array]` 的混合形式
* range : array_like, shape(2,2). optional 
* density : bool, optional
* normed : (deprecated) bool, optional. Use density instead.
* weights : array_like, shape(N,), optional. 对于每一个数据的权重数列.
* Return : 返回值
  * H : ndarray, shape(nx, ny).  
  * xedges ndarray, shape(nx+1,)
  * yedges ndarray, shape(ny+1,)


#### 3.16.2.3. bincount 原子统计

直方图的简化版本




## 3.17. Set 集合

### 3.17.1. unique

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

* numpy 的随机包这里独立的分一章  `numpy.random`
* 比 Pystl 的 random 包通用性更广, 值得学习

基本概念
1. BitGenerators    : 用于单纯的生成随机 bit 序列, 表现为随机32 or 64 bit 填入的 unsigned interger 
2. Generators       : 该库的用户接口, 将 BitGenerator 的随机序列转换为特定的概率分布以及范围的随机数
3. random generator: 主要负责将随机bit转为特定分布
4. RandomState: 已经被淘汰的旧版生成器, 只能基于一种 BitGenerators
   - RandomState 目前直接作为一个对象被放入了 该命名空间
   - 通过直接调用 `numpy.random` 下面的各种函数来使用默认的 generator 


使用流程:
1. 定义生成器 Generators 对象, 通过调用构造函数来定义不同 BitGenerators 的随机数生成器
2. 调用 Generator 对象的各种分布方法, 来获取具体的随机分布

## 4.1. Generator

基本生成器
* `default_rng(seed=None:{None, int, array_like[ints], SeedSequence, BitGenerator, Generator})`  : 
  * 被官方推荐的生成器, 默认使用 numpy 的 default BitGenerator `PCG64` , 效果优于 `MT19937`
  * seed : 接受多种 seed 的种类
  * return : The initialized generator object.
* `class numpy.random.Generator(bit_generator)` 标准的 generator 构造函数
  * bit_generator : BitGenerator 手动输入对应的 BitGenerator 

* `random.Generator.bit_generator`
  * generator 的一个类属性, 可以访问到该 generator 所使用的 BitGenerator 对象

## 4.2. Random Generation Function


通过使用生成器对象的方法可以产生任意区间和分布的随机数, 省略 `random.Generator.` 或者 `[TODO]`

### 4.2.1. Simple Random 简单的随机生成

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


### 4.2.2. Permutations 排列

* shuffle(x[, axis])
* permutation(x[, axis])

### 4.2.3. Distributions 分布函数

通用参数:
* `size` : int or tuple of ints, optional
  * 输出的随机 array 的 shape
  * 对于不同的分布有不同的参数指标, 例如高斯分布有 mean 和 var
    * numpy 的随机分布的参数也支持 array 输入
    * 如果分布的参数全都是标量而非 array, 整个分布函数最终输出单个数据
    * 否则 `np.broadcast(*pars).size` 会被使用


* `normal(loc=0.0, scale=1.0, size=None)`
  * 标准高斯分布
  * `loc`   : float or array_like of floats. 均值 Means
  * `scale` : float or array_like of floats. 标准差 std 而非方差, 非负
* `standard_normal(size=None, dtype=np.float64, out=None)`
  - 从标准的正态分布采样, 主要用来省略 std 和 mean 的输入
  - dtype : only float64 and float32 are supported.
* `uniform(low=0.0, high=1.0, size=None)`
  - 均一分布

# 5. Universal functions (ufunc)

经常出现在各种函数的参数中, 属于高级操作

A universal function (or ufunc for short) is a function that operates on ndarrays in an element-by-element fashion, supporting array broadcasting, type casting, and several other standard features. 


# NumPy's Module Structure

## numpy.lib

独立于 array routine 的一系列模组化的函数


### numpy.lib.stride_tricks

通过操作 strides 来实现类似于 sliding window 的效果

* as_strided : 原始函数, 需要谨慎操纵内存
* sliding_window_view : 滑动窗口生成, 不能指定步长



`lib.stride_tricks.sliding_window_view(x, window_shape, axis=None, *, subok=False, writeable=False)`
* 对于给定的 window 生成滑动窗口视图 `rolling or moving window`
* 参数 : 
  * x : array_like, 原始数据
  * window_shape: int or tuple of int. 滑动窗口的 shape, 如果不指定 axis 的话则长度必须 x.ndim 一致. int 输入等同于 `(i,)` 即不会进行扩张
  * axis : int or tuple of int, 如果指定的话需要于 window_shape 一样长
  * subok : bool, sub ok. 默认为 flase, 如果为 True 则子类将被传递? passed-through 不太懂 TODO
  * writeable : bool, 为 true 的时候才允许写入, 因为 view 本身有很多元素指向了相同的内存地址, 应当谨慎使用 


`lib.stride_tricks.as_strided(x, shape=None, strides=None, subok=False, writeable=True)`
* 更加低级的实现, 相当于直接操纵 array 的属性, 可以参照 ndarray.strides
* 不正确的计算会直接导致指向无效的内存引起程序崩溃
* shape : new array 的 shape, 默认与 x 相同
* strides : new array 的 strides, 默认与 x 相同

# 7. config

## 7.1. np.set_printoptions

1. 取消科学计数法显示数据 `np.set_printoptions(suppress=True)  `



2. 取消省略超长数列的数据 ` np.set_printoptions(threshold=sys.maxsize)` 需要 sys 包


### 7.1.1. numpy.shape

### 7.1.2. numpy.dot()  矩阵点乘

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




