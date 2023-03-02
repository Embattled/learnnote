- [1. numpy.array](#1-numpyarray)
  - [1.1. attributes](#11-attributes)
  - [1.2. scalars](#12-scalars)
  - [1.3. 元素运算](#13-元素运算)
    - [1.3.1. universal function](#131-universal-function)
    - [1.3.2. clip 修正边界](#132-clip-修正边界)
  - [1.4. Calculation 降维运算](#14-calculation-降维运算)
  - [1.5. Copies and Views](#15-copies-and-views)
  - [1.6. Indexing routines](#16-indexing-routines)
- [2. numpy other](#2-numpy-other)
  - [2.1. 内存排列规则](#21-内存排列规则)
- [3. Array objects](#3-array-objects)
  - [3.1. Standard array subclasses](#31-standard-array-subclasses)
- [4. Routines  array常规操作 API](#4-routines--array常规操作-api)
  - [4.1. Array creation](#41-array-creation)
    - [4.1.1. From shape or value](#411-from-shape-or-value)
    - [4.1.2. From existing data](#412-from-existing-data)
      - [4.1.2.1. From File](#4121-from-file)
      - [4.1.2.2. From Data](#4122-from-data)
      - [4.1.2.3. From Memory](#4123-from-memory)
    - [4.1.3. Numerical ranges 创建范围的array](#413-numerical-ranges-创建范围的array)
  - [4.2. Array manipulation 操纵更改 Array](#42-array-manipulation-操纵更改-array)
    - [4.2.1. Changing array shape 形态转换](#421-changing-array-shape-形态转换)
    - [4.2.2. Transpose-like operations 转置操作](#422-transpose-like-operations-转置操作)
    - [4.2.3. Changing number of dimensions 维度个数操作](#423-changing-number-of-dimensions-维度个数操作)
      - [4.2.3.1. expand\_dims 升维](#4231-expand_dims-升维)
      - [4.2.3.2. squeeze 压缩维度](#4232-squeeze-压缩维度)
    - [4.2.4. Joining arrays 拼接](#424-joining-arrays-拼接)
    - [4.2.5. Splitting arrays 拆分](#425-splitting-arrays-拆分)
    - [4.2.6. Tiling arrays](#426-tiling-arrays)
    - [4.2.7. Adding and removing elements 修改元素](#427-adding-and-removing-elements-修改元素)
      - [4.2.7.1. append](#4271-append)
      - [4.2.7.2. resize 强行更改 shape](#4272-resize-强行更改-shape)
    - [4.2.8. Rearranging elements 重新排列元素](#428-rearranging-elements-重新排列元素)
  - [4.3. Discrete Fourier Transform (numpy.fft)](#43-discrete-fourier-transform-numpyfft)
    - [4.3.1. Standard FFTs 标准傅里叶变换](#431-standard-ffts-标准傅里叶变换)
    - [4.3.2. Real FFTs 复数傅里叶变换](#432-real-ffts-复数傅里叶变换)
    - [4.3.3. Hermitian FFTs](#433-hermitian-ffts)
    - [Helper routines 辅助功能](#helper-routines-辅助功能)
  - [4.4. linalg](#44-linalg)
    - [4.4.1. SVD 奇异值分解](#441-svd-奇异值分解)
  - [4.5. numpy Input and Output  Numpy 数据的 IO](#45-numpy-input-and-output--numpy-数据的-io)
    - [4.5.1. Text Files](#451-text-files)
  - [4.6. Linear algebra 线性代数计算](#46-linear-algebra-线性代数计算)
    - [4.6.1. Matrix and vector products 向量矩阵乘法](#461-matrix-and-vector-products-向量矩阵乘法)
      - [4.6.1.1. 矩阵乘法](#4611-矩阵乘法)
      - [einsum](#einsum)
    - [4.6.2. Solving equations and inverting matrices 计算矩阵方程或者逆](#462-solving-equations-and-inverting-matrices-计算矩阵方程或者逆)
  - [4.7. Logic functions 逻辑计算](#47-logic-functions-逻辑计算)
    - [4.7.1. Truth value testing](#471-truth-value-testing)
  - [4.8. Masked array operations](#48-masked-array-operations)
  - [4.9. Mathematical function 数学操作](#49-mathematical-function-数学操作)
    - [4.9.1. Trigonometric functions 三角函数](#491-trigonometric-functions-三角函数)
    - [4.9.2. Hyperbolic functions 双曲线函数](#492-hyperbolic-functions-双曲线函数)
    - [4.9.3. Rounding 最近值](#493-rounding-最近值)
    - [4.9.4. Sums, products, differences 求和求积求差](#494-sums-products-differences-求和求积求差)
    - [4.9.5. Exponents and logarithms 指数](#495-exponents-and-logarithms-指数)
    - [4.9.6. Rational routines 最大公因数 最小公倍数](#496-rational-routines-最大公因数-最小公倍数)
    - [4.9.7. Extrema Finding 极值寻找](#497-extrema-finding-极值寻找)
    - [4.9.8. 杂项](#498-杂项)
      - [4.9.8.1. convolve 卷积](#4981-convolve-卷积)
      - [4.9.8.2. clip 裁剪](#4982-clip-裁剪)
      - [4.9.8.3. interp 简易线性插值](#4983-interp-简易线性插值)
  - [Padding Arrays](#padding-arrays)
  - [Polynomials 多项式](#polynomials-多项式)
  - [4.10. Sorting, Searching, Counting 排序 搜索 计数](#410-sorting-searching-counting-排序-搜索-计数)
    - [4.10.1. Sorting 排序](#4101-sorting-排序)
      - [4.10.1.1. 基础排序](#41011-基础排序)
      - [4.10.1.2. 部分有序](#41012-部分有序)
    - [4.10.2. Searching 元素查找](#4102-searching-元素查找)
      - [4.10.2.1. 最大值选择](#41021-最大值选择)
      - [4.10.2.2. 逻辑选择值 where](#41022-逻辑选择值-where)
      - [4.10.2.3. 非零选择](#41023-非零选择)
  - [4.11. Statistics 统计](#411-statistics-统计)
    - [4.11.1. Averages and variances 平均和方差](#4111-averages-and-variances-平均和方差)
    - [4.11.2. Histograms](#4112-histograms)
      - [4.11.2.1. histogram 一维数据直方图](#41121-histogram-一维数据直方图)
      - [4.11.2.2. histogram2d 二维直方图](#41122-histogram2d-二维直方图)
      - [4.11.2.3. bincount 原子统计](#41123-bincount-原子统计)
  - [4.12. Set 集合](#412-set-集合)
    - [4.12.1. unique](#4121-unique)
- [5. numpy.random](#5-numpyrandom)
  - [5.1. Generator](#51-generator)
  - [5.2. Random Generation Function](#52-random-generation-function)
    - [5.2.1. Simple Random 简单的随机生成](#521-simple-random-简单的随机生成)
    - [5.2.2. Permutations 排列](#522-permutations-排列)
    - [5.2.3. Distributions 分布函数](#523-distributions-分布函数)
- [6. Universal functions (ufunc)](#6-universal-functions-ufunc)
- [7. numpy 常规功能](#7-numpy-常规功能)
  - [7.1. numpy 的IO](#71-numpy-的io)
    - [7.1.1. 类型转换](#711-类型转换)
    - [7.1.2. numpy binary files](#712-numpy-binary-files)
    - [7.1.3. text file](#713-text-file)
- [8. config](#8-config)
  - [8.1. np.set\_printoptions](#81-npset_printoptions)
    - [8.1.1. numpy.shape](#811-numpyshape)
    - [8.1.2. numpy.dot()  矩阵点乘](#812-numpydot--矩阵点乘)

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

# 2. numpy other


## 2.1. 内存排列规则

order{‘K’, ‘A’, ‘C’, ‘F’}, optional

# 3. Array objects 

## 3.1. Standard array subclasses



# 4. Routines  array常规操作 API

对 Array 数据的各种常规操作

* 尽量明确各个函数的返回值是 copy 的还是只是映射

函数可以分为
1. 类方法和全局函数
2. 返回改变后的值和改变自身(inplace)


## 4.1. Array creation

numpy.array 的各种创建函数能够创建各种各样的预设 array

* 所有函数都有 `dtype` 参数
* 所有函数都有 `order` 参数
* 部分函数里有 `like`  参数 : array_like, optional, 用于将返回值创建成 np.array 以外的数据类型


### 4.1.1. From shape or value

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


### 4.1.2. From existing data

从既存的数据中创建一个 array, 某种程度上也算是 numpy 的文件 Input

#### 4.1.2.1. From File 

`numpy.fromfile(file, dtype=float, count=- 1, sep='', offset=0, *, like=None)`  文件读取
* `file`  : Open file object or str or Path, 1.17.0 pathlib.Path objects are now accepted.
* `dtype` : 用于指定 array 类型, 同时对于 binary 文件还决定了读取的步长 (即单个 item 的大小)
* `count` : 用于指定要读取的 item 的数量, -1代表读完整个文件
* `sep`   : 指定了该文件是否是 binary 或者 text file, 默认是空字符代表了二进制文件, 如果是空格分隔符 ` `, 则代表 text 文件, 同时分割匹配符会匹配1到多个空白字符
* `offset`: 读取时候的向后偏移, 只在 binary 的时候起作用

#### 4.1.2.2. From Data

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

#### 4.1.2.3. From Memory

`numpy.frombuffer(buffer, dtype=float, count=- 1, offset=0, *, like=None)`
* 从一个 bytes 数据中读取数据
* dtype : 指定数据的类型
* count : 指定按照 dtype 的类型读取的数据个数
* 该函数默认不进行拷贝


`numpy.from_dlpack(x, /)`
* 把一个 DLPACK 数据转换成 ndarray
* 所谓 DLPACK 数据即满足 `__dlpack__ ` protocol 的数据
  * A Python object that implements the __dlpack__ and __dlpack_device__ methods. 



### 4.1.3. Numerical ranges 创建范围的array

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
  * 从坐标向量生成坐标 mesh, 该函数主要用于作图等其他函数的输入数据生成  
  * *xi, 可以是任意维度的输入, 代表 N-D 的坐标
  * `indexing` : {‘xy’, ‘ij’}, optional. 用于指定坐标的先后顺序
    * xy 代表了图像领域的坐标, x 是横坐标, y 是纵坐标, 但是在数据索引上 y 纵坐标是行数因此靠前, 即坐标顺序是颠倒的
    * ij 代表了普通数列的索引方式, 靠前的维度坐标也靠前
  * return : X1, X2, ...,XN  根据 len(xi) 和 indexing
    * 代表了该空间下每个点的具体坐标数列  
    * 点的坐标依次是 `( x1[0],x2[0],...,xn[0]  ) , (x1[1],x2[1],..,xn[1])`

## 4.2. Array manipulation 操纵更改 Array

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

### 4.2.1. Changing array shape 形态转换

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


### 4.2.2. Transpose-like operations 转置操作

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



### 4.2.3. Changing number of dimensions 维度个数操作

对 array 的维度 (而不是shape)进行操作的函数

atleast_函数集: 注意该输入是 `*arry` 即多参数转化成元组输入, 如果输入多个参数, 则这些参数会被转化成对应维度的 array
* `atleast_1d(*arys)`
* `atleast_2d(*arys)`
* `atleast_3d(*arys)`
* return ndarray

#### 4.2.3.1. expand_dims 升维

朴素升降维:
* `numpy.expand_dims(a, axis)`
  * axis : int or tuple of ints. 范围是 (0,a.ndim)
  * 被插入的维度的位置 == axis , 意思是其后的维度会被顺延
  * 被插入的维度的 shape 都是 1
  * return : `View` of a with the number of dimensions increased. 因为不需要更改内存分配

1. `arr=arr[:,:,np.newaxis`
2. `arr=np.array([arr])`


#### 4.2.3.2. squeeze 压缩维度 
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

### 4.2.4. Joining arrays 拼接

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

### 4.2.5. Splitting arrays 拆分 

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

### 4.2.6. Tiling arrays

平铺一个 array 

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


### 4.2.7. Adding and removing elements 修改元素

将 np.ndarray 以类似于普通 list 的视角操作

#### 4.2.7.1. append

该函数不存在 in-place 模式, Append values to the end of an array.

* `numpy.append(arr, values, axis=None)` 
  * arr : Values are appended to a `copy` of this array.
  * values: array_like. 维度 ndim 必须和 arr 一样
  * axis : None 要被插入的维度
    * values : 必须是 correct shape (the same shape as arr, excluding axis).
    * if axis = None, values can be any shape and will be flattened before use
    * If axis is None, out is a flattened array.
  * return : A copy of arr with values appended to axis. 
  

#### 4.2.7.2. resize 强行更改 shape

不同于 reshape 的resize
* 会带有填充以及裁剪的更改 array 形态
* 以 C-order进行
* `numpy.resize(a, new_shape)` : 不足的地方用 a 的重复来填充, 返回新的 array
* `ndarray.resize(new_shape, refcheck=True)` in-place 的修改, 不足的地方会用 0 填充
  * 如果 refcheck 是True, 那么如果该 array 有别的引用的话会发生 raise 来停止该操作
* 


### 4.2.8. Rearranging elements 重新排列元素  

最经典的 reshape 也被包括在这里, 但是上面写了这里就省略

flip 以及 懒人 flip : Reverse the order of elements in an array along the given axis
* `numpy.flip(m, axis=None)`  
  * axis : None or int or tuple of ints, optional. 如果 axis 是默认值 None, 那么所有维度都会一起反转
* `numpy.flipud(m)`
  * 反转上下, 即行, axis = 0
* `numpy.fliplr(m)`
  * 反转左右, 列, 要求输入数据必须是 2维以上, axis=1


## 4.3. Discrete Fourier Transform (numpy.fft)

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


### 4.3.1. Standard FFTs 标准傅里叶变换

### 4.3.2. Real FFTs 复数傅里叶变换

### 4.3.3. Hermitian FFTs 


### Helper routines 辅助功能

* `np.fft.fftshift(x, axes=None)`: 便于 fft 结果的可视化
  * 标准 np.fft 运算的结果, 0频率的部分位于结果数列 `x[0]`, 然而一般为了可视化, 更加倾向于将 0 频率的部分移到数组中间
  * `axes=None`,  int or shape tuple, optional, 很重要, 需要同 fft 应用时候的 axis 参数一致
  * 注意, 最终 shift 的结果上, `y[0]` 是 Nyquist component only if `len(x)` is even.

* `np.fft.ifftshift(x, axes=None)` : fftshift 的逆运算
  * identical for even-length x
  * differ by one sample for odd-length x


## 4.4. linalg 

### 4.4.1. SVD 奇异值分解

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

## 4.5. numpy Input and Output  Numpy 数据的 IO

numpy 对于各种类型的输出支持的很好, 要注意对于 pandas 的 DataFrame 支持写在了别的段里  

通用参数: 
* fmt : str or sequence of strs, optional
  * 用于指定数据在输出时候的格式
  * 可能在别的地方有完整的文档, 保留为 [TODO]


### 4.5.1. Text Files

保存 save :
* `numpy.savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)`
  * 将一个 array 保存到 txt
  * X : array , 只支持 1D 或者 2D
  * delimiter : 2D 数据的时候分割列的符号
  * newline : 1D 数据或者 2D数据分割行 的符号
  * header  : 很方便, 在参数里就能直接传入数据的首列标识符, 还会贴心的加上 `#`
  * footer  : 同理, 添加到文件末尾  
  * comments: 会被添加到 header 和 footer 








## 4.6. Linear algebra 线性代数计算 

包含了 numpy 的各种线性代数计算函数, 其中一些函数定义在子包 `numpy.linalg` 中  
* numpy 的线性代数函数基于各种 BLAS 和 LAPACK 库来实现, 提供了高速的各种线性代数计算
* 对于性能要求严格的使用场景, numpy 官方更加推荐使用环境依存的线性代数库, e.g. OpenBLAS, MKL, ATLAS, 这些库提供了对多线程和多核心的完整支持
* 对于 `SciPy`库, 也提供了对应的 线性代数部分 `scipy.linalg`, 和 numpy 有一定的重合
  * pros scipy 的库更加全面, 有一些 numpy 中没有的计算函数, 对于重合的函数, scipy 也有一些额外的参数
  * cons numpy 的有些函数对于 array 的 broadcasting 的效果更好

### 4.6.1. Matrix and vector products 向量矩阵乘法


#### 4.6.1.1. 矩阵乘法
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

#### einsum  

评估操作数的爱因斯坦求和约定
Evaluates the Einstein summation convention on the operands.

爱因斯坦求和约定, 可以用一个简单的方式表示许多常见的 多维线性数据运算.  
在 numpy 中
* 隐式implicit模式下, 直接计算对应的值
* 显式explicit模式下, provides further flexibility to compute other array operations that might not be considered classical Einstein summation operations.


`numpy.einsum(subscripts, *operands, out=None, dtype=None, order='K', casting='safe', optimize=False)`

### 4.6.2. Solving equations and inverting matrices 计算矩阵方程或者逆

* `numpy.linalg.inv(a)`  计算一个矩阵的逆
  * 具体在代码上表现为 `dot(a, ainv) = dot(ainv, a) = eye(a.shape[0])`  
  * 注意, 这个 a 是支持多维的, 只需要满足 `(...,M,M)` 即可, 自动使用后两维作为矩阵
  * 返回值 ainv 也满足 shape `(...,M,M)`


## 4.7. Logic functions 逻辑计算

包括 ndarray 之间的逻辑运算以及自身元素的检查逻辑

### 4.7.1. Truth value testing

只有两个函数
* `numpy.all(a, axis=None, out=None, keepdims=<no value>, *, where=<no value>)` 是否全部为 True
* `numpy.any(a, axis=None, out=None, keepdims=<no value>, *, where=<no value>)` 是否有 True

## 4.8. Masked array operations 

同 Logic 操作非常相似, 主要是通过各种逻辑判断来生成 mask 数据  



## 4.9. Mathematical function 数学操作

绝大多数常用的数学基础函数都属于该分类

### 4.9.1. Trigonometric functions 三角函数

### 4.9.2. Hyperbolic functions 双曲线函数



### 4.9.3. Rounding 最近值

### 4.9.4. Sums, products, differences 求和求积求差

* sum()
* cumsum(a) 累加求和

### 4.9.5. Exponents and logarithms 指数
### 4.9.6. Rational routines 最大公因数 最小公倍数

### 4.9.7. Extrema Finding 极值寻找

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

### 4.9.8. 杂项

有时候需要的很特殊的功能, 没办法分类, 目前学习的有

* convolve
* clip
* interp

#### 4.9.8.1. convolve 卷积

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

#### 4.9.8.2. clip 裁剪

`numpy.clip(a, a_min, a_max, out=None, **kwargs)`
* 裁剪一个 array, 比用最大最小值实现要快, 且代码更清晰
* Equivalent to but faster than `np.minimum(a_max, np.maximum(a, a_min))`
* a_min, a_max : 可以指定成 None, 表示为没有任何检查, 只能有一个被指定成 None
  * 注意 a_min a_max 是没有大小比较验证的, 需要用户自己保证


#### 4.9.8.3. interp 简易线性插值

`numpy.interp(x, xp, fp, left=None, right=None, period=None)`  
One-dimensional linear interpolation for monotonically increasing sample points.
* 限制比较多, 只支持 1D, 且数列需要单调递增, 文档建议使用 scipy 的相关插值函数
* x ： 要进行插值的坐标
* fp : 被插值的数列,
* xp : fp 各个元素在 x 轴上的具体坐标, 必须是单调递增的, 除非指定 period, 不能包含 NaN.
* left, right. 用于指定 x 在超越界限 (大于 `x[-1]` 小于 `x[0]` ) 的时候的输出值
* period : A period for the x-coordinates, xp 的周期, 一般用来计算角度, 即360 度为一圈, 720度会被正确的放在 0 度的位置


## Padding Arrays

numpy 的填充函数, 只有一个函数单独作为了一类, 可以对任意维度进行填充

`numpy.pad(array, pad_width, mode='constant', **kwargs)`  
* array : array, array_like of rank N
* pad_width: {sequence, array_like, int}
* mode : str, 用于表示填充操作的种类
  * 


## Polynomials 多项式

numpy 1.4 引进的多项式包, 是对于之前的函数包 `numpy.poly1d` 的扩展  

概念区分:
* `polynomial module` 指的是 old API, 定义在了 `numpy.lib.polynomial` 里, 其中包括
  * 旧的多项式类 `numpy.poly1d`
  * 其他的 polynomial functions
* `polynomial package` 值得是 new API, 定义在了 `numpy.polynomial`
  * convenience classes for the different kinds of polynomials



## 4.10. Sorting, Searching, Counting 排序 搜索 计数

这里的 counting 都是很简单的函数, 更详细的统计在 statistics 模块


### 4.10.1. Sorting 排序

#### 4.10.1.1. 基础排序

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


#### 4.10.1.2. 部分有序

* partition(a, kth[, axis, kind, order])    :  Return a partitioned copy of an array.
* argpartition(a, kth[, axis, kind, order]) : 

### 4.10.2. Searching 元素查找

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

#### 4.10.2.1. 最大值选择

* `argmax(a[, axis, out, keepdims])`
  * Returns the indices of the maximum values along an axis.
  * 一般用于机器学习通过可信度得到最终 label
* `argmin(a[, axis, out, keepdims])`
  * Returns the indices of the minimum values along an axis.



#### 4.10.2.2. 逻辑选择值 where

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



#### 4.10.2.3. 非零选择

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


## 4.11. Statistics 统计

更加完整的统计函数定义在了这里

### 4.11.1. Averages and variances 平均和方差

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


### 4.11.2. Histograms


直方图统计, 在 Statistic 分类的函数中属于一个大类, 不止一个函数  

* numpy.histogram    : 普通一维直方图统计, 


#### 4.11.2.1. histogram 一维数据直方图
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

#### 4.11.2.2. histogram2d 二维直方图

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


#### 4.11.2.3. bincount 原子统计

直方图的简化版本




## 4.12. Set 集合

### 4.12.1. unique

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




# 5. numpy.random

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

## 5.1. Generator

基本生成器
* `default_rng(seed=None:{None, int, array_like[ints], SeedSequence, BitGenerator, Generator})`  : 
  * 被官方推荐的生成器, 默认使用 numpy 的 default BitGenerator `PCG64` , 效果优于 `MT19937`
  * seed : 接受多种 seed 的种类
  * return : The initialized generator object.
* `class numpy.random.Generator(bit_generator)` 标准的 generator 构造函数
  * bit_generator : BitGenerator 手动输入对应的 BitGenerator 

* `random.Generator.bit_generator`
  * generator 的一个类属性, 可以访问到该 generator 所使用的 BitGenerator 对象

## 5.2. Random Generation Function


通过使用生成器对象的方法可以产生任意区间和分布的随机数, 省略 `random.Generator.` 或者 `[TODO]`

### 5.2.1. Simple Random 简单的随机生成

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


### 5.2.2. Permutations 排列

* shuffle(x[, axis])
* permutation(x[, axis])

### 5.2.3. Distributions 分布函数

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

# 6. Universal functions (ufunc)

经常出现在各种函数的参数中, 属于高级操作

A universal function (or ufunc for short) is a function that operates on ndarrays in an element-by-element fashion, supporting array broadcasting, type casting, and several other standard features. 



# 7. numpy 常规功能

## 7.1. numpy 的IO

numpy 的数据IO可以简单分3类:
* 二进制IO
* txt IO
* 与python 内置 string 的转换

numpy 的 IO 也一定程度上基于 pickle, 具有一定的不安全性

通用参数:
* file : file-like object, string, or pathlib.Path
  
### 7.1.1. 类型转换

在 numpy 官方文档中, ndarray 相关的类型转换也被归纳为 IO 的一部分

* ndarray.tolist()
  * 将 np array 转换成python自带的列表, 该函数是拷贝的
  * 会把元素的类型也转成 python 自带的数字类型
  * 对于 0-dim 的 array `np.array(1)`, 直接使用 list(a) 会报异常, 只能用 tolist()
  * 相比与 list(a), a.tolist() 更加安全, 且附带了类型转换
* ndarray.tofile(fid[, sep, format])


### 7.1.2. numpy binary files

最基础的保存方法, 因为是二进制的, 所以最好只通过 numpy 访问, 文件后缀为 `.npy`

* load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
* save((file, arr, allow_pickle=True, fix_imports=True))
* savez
* savez_compressed

### 7.1.3. text file

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

# 8. config

## 8.1. np.set_printoptions

1. 取消科学计数法显示数据 `np.set_printoptions(suppress=True)  `



2. 取消省略超长数列的数据 ` np.set_printoptions(threshold=sys.maxsize)` 需要 sys 包


### 8.1.1. numpy.shape

### 8.1.2. numpy.dot()  矩阵点乘

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




