# 1. numpy 

python数值包 最基础的列表处理包 被其他许多包所依赖  

# 2. numpy.array
NumPy’s array class is called `ndarray`. It is also known by the alias `array`.  
  
* numpy最基础的类, 所有元素都是同一类型
* 和 python 自带的 `array.array` 是不同的  
* python 自带的类只能处理一维数组, 并且函数很少

```py
A = np.array([[1,3,4], [2,3,5], [1,2,3], [5,4,6]])
print(A)
'''
[[1 3 4]
 [2 3 5]
 [1 2 3]
 [5 4 6]]
'''
```
## 2.1. attributes

* ndim : 维度
* shape :  A tuple of integers indicating the size of the array in each dimension.
* size :  Total number of elements of the array. This is equal to the product of the elements of `shape`.
* dtype : 元素类型
* itemsize : 元素大小 bytes
* data : 指向实际数据的 `array` 一般不需要使用

## 2.2. creation

1. From a regular Python list or tuple using the array function
   * 必须以序列的形式传入第一个参数第
   * 元素类型会被自动推导, 或者用`dtype=`指定
2. 使用基础矩阵生成函数, 也可以指定类型
   * one()
   * zero()
   * empty()  不确定值, 取决于该块内存原本的值
   * 第一参数为元组, 指定维度大小
3. 使用 `arange` 生成等差整数序列
   * 使用方法基本和 内置函数 range 相同
   * python 内置函数 range 返回的是普通 array, 该方法返回的是 np.array
4. 使用 `linspace` 创建序列
   * 同 arange 相同, 只不过思想改成了分割
   * linspace(起始值,结束值,分割成多少个值)

## 2.3. 基础运算

两个类型不同的 np.array 运算, 结果元素类型会是更通用的或者更精确的一方  
numpy的二元基础运算都是元素层面的  
* `+ - ** *`
* `*`乘法也是元素层面, 两个矩阵使用 `*`  不会进行矩阵乘法
* 组合运算符 `+= -=` 会进行直接结果替换
* 使用矩阵乘法可以使用 `@` 运算符
* 使用矩阵乘法也可以用 .dot() 方法


单元运算也有实现
* .sum 返回全元素的和
* .min 
* .max
* cumsum 返回维度相同的累加矩阵  
* 通过指定以上函数的 `axis=i` 参数, 可以指定运算所执行的对象
    * `i`从0开始, 指代最外层的括号, 即索引访问时最左边的方括号
    * 遍历`i`下标可能的每个值, 得到一个结果, 结果为 


### 3.1.2. numpy.dot()  矩阵点乘

np.diag(s)  将数组变成对角矩阵  
使用numpy进行矩阵乘法   

```py
# 使用svp分解矩阵
U, s, Vh = np.linalg.svd(A, full_matrices=False)
#使用 .dot() 将原本的矩阵乘回来 
Us = np.dot(U, np.diag(s))
UsVh = np.dot(Us, Vh)

```

# 3. config

## 3.1. np.set_printoptions

1. 取消科学计数法显示数据 `np.set_printoptions(suppress=True)  `
2. 取消省略超长数列的数据 ` np.set_printoptions(threshold=sys.maxsize)`



### 3.1.1. numpy.shape


```py

# 计算一个矩阵的奇异值
A = np.array([[1,3,4], [2,3,5], [1,2,3], [5,4,6]])
U, s, Vh = np.linalg.svd(A, full_matrices=False)

print(np.shape(U), np.shape(s), np.shape(Vh))

'''输出
(4, 3) (3,) (3, 3)
'''
```




## 3.2. linalg 

### 3.2.1. SVD 奇异值分解

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