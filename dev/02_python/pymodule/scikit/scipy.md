# 1. scipy

Scipy是一个用于数学、科学、工程领域的常用软件包，可以处理插值、积分、优化、图像处理、常微分方程数值解的求解、信号处理等问题。  
它用于有效计算Numpy矩阵，使Numpy和Scipy协同工作，高效解决问题。  

| 模块名            | 应用领域           |
| ----------------- | ------------------ |
| scipy.cluster     | 向量计算/Kmeans    |
| scipy.constants   | 物理和数学常量     |
| scipy.fft          | 傅立叶变换         |
| scipy.integrate   | 积分程序           |
| scipy.interpolate | 插值               |
| scipy.io          | 数据输入输出       |
| scipy.linalg      | 线性代数程序       |
| scipy.ndimage     | n维图像包          |
| scipy.odr         | 正交距离回归       |
| scipy.optimize    | 优化               |
| scipy.signal      | 信号处理           |
| scipy.sparse      | 稀疏矩阵           |
| scipy.spatial     | 空间数据结构和算法 |
| scipy.special     | 一些特殊的数学函数 |
| scipy.stats       | 统计               |

# 2. Fourier Transforms ( scipy.fft )

因为是计算机的数值计算库, 所以天然是离散的  
定义了很多 离散傅里叶变换 discrete Fourier transform (DFT) 的实现  

The DFT has become a mainstay of numerical computing in part because of a very fast algorithm for computing it, called the Fast Fourier Transform (FFT)

同为正交变换的离散正余弦变换也在该库中, 还有 Hankel Transforms  


## 2.1. Fast FFourier Transforms 快速傅里叶

函数命名规则 `(i)(r/h)fft(n)`
* i : 是否是逆变换
* r/h: 数列的形式,  r 复数数列, h Hermitian complex array.
* n : 针对几维数据的变换  支持 2,n

函数通用输入参数:
* `x` array_like  : 输入的数列, 支持 complex
* `n/s=None` int/list : 数值要进行变换的长度, 即输出的结果数列的 shape, 对于多维FFT, len(s)=x.ndim, 而根据 n/s 具体的数值会对输入数列进行裁剪或者 0 padding
* `axis=-1`   : 要进行 FFT 的维度, 影响最终输出的 ndim, 默认是针对全部数据的 FFT, 对于 fftn, 也会受 s 参数的影响
* 

## 2.2. Discrete Sin and Discrete Cosine Transforms (DST DCT)

## 2.3. Fast Hankel Transforms

## 2.4. Helper functions 辅助函数

# 3. Interpolation (scipy.interpolate)

scipy 的线性插值包, 大概可以分为:
* 单变量插值
* 多变量插值
* 1-D Splines
* 2-D Splines
* Additional tools

同系列的包管理方法一样, 各种功能具有 类 和 单函数两种实现  


## 3.1. Multivariate interpolation 

### 3.1.1. 网格数据 (griddata)

`scipy.interpolate.griddata(points, values, xi, method='linear', fill_value=nan, rescale=False)`
* 无限制 任意维度网格插值
* Interpolate unstructured D-D data.
* points : 原始数据点的坐标, 令 数据的维度是 D, 点的个数是 n
  * points 接受 2-D ndarray 输入, 维度为 (n,D), 代表 n 个点的坐标
  * 接受 tuple 输入, len(points) = D, 每个元素是每个维度的坐标1-D np.array
* values : Data values. shape (n,). 每个点的数据值
* xi : 要被插值的点的坐标, 令 插值点的个数为 m
  * 接受的两种输入格式同 points 一致
* method : Method of 插值
  * nearest
  * linear
  * cubic 

### 3.1.2. Regular Grid


## 3.2. Spline interpolation

样条插值, 是区别于多项式插值的另一个插值算法家族, Spline interpolation 于2018年成为正式的计算机科学技术名词  
* 与多项式插值不同, 产生的结果是一个分段函数 (分段多项式函数 piecewise polynomial), 降低了结果函数的整体性, 但是有效避免了 龙格现象(Runge Phenomenon)
* 同时对于每段插值, 即使 degree 阶很小, 也能活得很好的效果
* 样条在 计算机辅助设计和 计算机图形学 领域中被广泛应用  


原文描述
* a spline is a special function defined piecewise by polynomials
* instead of fitting a single, high-degree polynomial to all of the values at once, spline interpolation fits low-degree polynomials to small subsets of the values


Spline interpolation 的基础步骤
1. 计算曲线的样条表示
2. 评估样条曲线在所需要的点的效果


# 4. Multidimensional image processing ( scipy.ndimage ) 

Scipy 也有图像处理的相关函数, 对比 scikit-image 里较为基础, 但也足够强大  


## 4.1. Filters - 通用滤波


## 4.2. Fourier filters - 傅里叶滤波

## 4.3. Interpolation - 插值


# 5. Signal Processing (scipy.signal)

SciPy 中专门用于处理信号的包
* 包括了一些 filter
* 一些 filter design tools
* 一些 B-spline interpolation algorithms for 1- and 2-D data

## 5.1. Filtering

Filtering : a generic name for any system that modifies an input signal in some way.

在 SciPy 中, signal 也是 Numpy array  


## 5.2. B-splines

# 6. scipy.sparse  Sparse Arrays

稀疏矩阵的处理函数, 可以高效的处理
* 线性代数 `scipy.sparse.linalg`
* 基于图的计算 `graph-based computations`

对应的, 稀疏矩阵格式的缺点是, 没办法灵活的进行
* slicing, reshaping, assignment

## 6.1. Sparse Arrays User Guide 

https://docs.scipy.org/doc/scipy/tutorial/sparse.html

## 6.2. Sparse array classes 稀疏数列


`class dok_array(arg1, shape=None, dtype=None, copy=False)`
* Dictionary Of Keys based sparse array.
* 基于 keys 的 稀疏 array 字典, 主要用于增量构建稀疏数组
* 允许 O(1) 的元素访问
* 不允许 keys 重复
* 可以方便的转换为 coo_array
* 参数:
  * args1 : 用于初始化 dok 的数据或者指定 dok 的维度
    * `dok_array(D)` : D 是 2维 ndarray
    * `dok_array(S)` : S 是另一个 matrix 或者 array, 相当于执行 `S.todok()`
    * `dok_array((M,N), [dtype])` : 通过指定维度来初始化 dok 的 shape


## 6.3. Sparse matrix classes 稀疏矩阵

` class coo_array(arg1, shape=None, dtype=None, copy=False)`




`class csr_matrix(arg1, shape=None, dtype=None, copy=False)`
* 参数:
  * 


## 6.4. scipy.sparse.csgraph 压缩稀疏图的管线 Compressed sparse graph routines

一个有 N 个节点的图可以用 (NxN) 的连接矩阵表达连接关系, 以及每一个边的权重

在 scipy 中
* dense array representations : 无连接的边表示为 `G[i,j]=0, infinity, NaN`
* dense masked rpresentations : 无连接的边表示为 指定的 masked values. 这对于边的权重存在 0 值的应用场景很有用
* sparse array representations : 无连接的边表示为 non-entries. 这种 sparse 表示方法直接允许 0 权重的边


```py
# 从 mask array 转换 sparse matrix 来表示稀疏图
G_dense = np.array([[0, 2, 1],

                    [2, 0, 0],

                    [1, 0, 0]])

G_masked = np.ma.masked_values(G_dense, 0)
from scipy.sparse import csr_matrix
G_sparse = csr_matrix(G_dense)

# 带 mask 的表达, 此时边的权重可以为0, 如果使用 csr_matrix 会报错
G2_data = np.array([[np.inf, 2,      0     ],

                    [2,      np.inf, np.inf],

                    [0,      np.inf, np.inf]])

G2_masked = np.ma.masked_invalid(G2_data)
from scipy.sparse.csgraph import csgraph_from_dense
G2_sparse = csgraph_from_dense(G2_data, null_value=np.inf)
```


`minimum_spanning_tree(csgraph, overwrite=False)`
* 执行 kruskal 最小联通树 算法, 输入数据是 无向图
* 参数:
  * csgraph: array_like or sparse matrix, 2 dimension. 二维连通图
  * overwrite: bool, 是否直接覆盖整个图, 提高运算效率. 
* 返回值:
  * span_tree: csr_matrix, NxN 的稀疏表达, 表示最终的最小生成树


# 7. Statistics ( scipy.stats ) 
 
### 7.0.1. 偏度（skewness） 和 峰度（peakedness；kurtosis）又称峰态系数
* 偏度（skewness）
  * 是统计数据分布偏斜方向和程度的度量，是统计数据分布非对称程度的数字特征。  
  * 偏度定义中包括正态分布（偏度=0），右偏分布（也叫正偏分布，其偏度>0），左偏分布（也叫负偏分布，其偏度<0）
* 峰度（peakedness；kurtosis）又称峰态系数
  * 表征概率密度分布曲线在平均值处峰值高低的特征数。
  * 直观看来，峰度反映了峰部的尖度。随机变量的峰度计算方法为：随机变量的四阶中心矩与方差平方的比值。
  *  If the kurtosis is greater than 3, then the dataset has heavier tails than a normal distribution (more in the tails).
  *  If the kurtosis is less than 3, then the dataset has lighter tails than a normal distribution (less in the tails). 

```py
variable = player_table['Height(inches)']

# 使用scipy计算偏度（skewness）
s = skew(variable)
print(f'Skewness {s}')

# 使用pandas计算
s=variable.skew()
print(f'Skewness {s}')

# 计算 Z 和 P
zscore, pvalue = skewtest(variable)
print(f'z-score {zscore}')
print(f'p-value {pvalue}')


# 使用pandas计算峰度
k = kurtosis(variable)
zscore, pvalue = kurtosistest(variable)

print(f'Kurtosis {k}')
print(f'z-score {zscore}')
print(f'p-value {pvalue}')

```


###  7.0.2. T-Test

```py

from scipy.stats import ttest_ind

group1 = player_table["Position"] == 'Catcher'
group2 = player_table["Position"] == 'Outfielder'
variable = player_table["Height(inches)"]

t, pvalue = ttest_ind(variable[group1], variable[group2],axis=0, equal_var=False)

print(f"t statistic {t}")
print(f"p-value {pvalue}")

```

### 7.0.3. ANOVA

```py
from scipy.stats import f_oneway

group1 = player_table["Position"] == 'Catcher'
group2 = player_table["Position"] == 'Outfielder'
group3 = player_table["Position"] == 'Shortstop'
group4 = player_table["Position"] == 'Starting Pitcher'

variable = player_table["Height(inches)"]

f, pvalue = f_oneway(variable[group1],variable[group2],variable[group3],variable[group4])

print(f"One-way ANOVA F-value {f}")
print(f"p-value {pvalue}")

```

### 7.0.4. from  chi-square statistic

scipy.stats import chi2_contingency

```py

pcts = [0, .25, .5, .75, 1]
players_binned = pd.concat(
  [pd.qcut(player_table.iloc[:,3], pcts, precision=1),
  pd.qcut(player_table.iloc[:,4], pcts, precision=1),
  pd.qcut(player_table.iloc[:,5], pcts, precision=1)],
  join='outer', axis=1)


from scipy.stats import chi2_contingency

table = pd.crosstab(player_table['Position'], players_binned['Height(inches)'])
print(table)

chi2, p, dof, expected = chi2_contingency(table.values)
print(f'Chi-square {chi2}   p-value {p}') 


```

### 7.0.5. pearson
a = player_table["Height(inches)"]
b = player_table["Weight(lbs)"]
#print(a)
rho_coef, rho_p = spearmanr(a, b)
r_coef, r_p = pearsonr(a, b)

print(f'Pearson r: {r_coef}')
print(f'Spearman r: {rho_coef}')

# 8. Multidimensional image processing ( scipy.ndimage )


多维度图像处理包


## Filters - 图像滤波



`gaussian_filter(input, sigma, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0, *, radius=None, axes=None)`
* 参数
  * input: array_like
  * sigma : scalar, sequence of scalars 用于指定高斯核的标准差
  * mode: str or sequence. 用于指定输入图在边界上的拓展方法
    * `reflect` (d c b a | a b c d | d c b a)
    * `constant` (k k k k | a b c d | k k k k)
    * `nearest` (a a a a | a b c d | d d d d)
    * `mirror` (d c b | a b c d | c b a)
    * `wrap` (a b c d | a b c d | a b c d)
  * axes : tuple of int or None, 指定执行 filter 的维度. 默认值 None 会导致滤波在所有维度上执行
    * 如果 axes 指定, 则 `sigma, order, mode, radius` 必须同 axes 的长度相同
  * radius : None or int or sequence of ints, 高斯核的尺寸
    * 默认值 None 会采用 `radius = round(truncate * sigma)`
  * truncate : float, 默认值 4.0 
    * 标准差截断值, 用于生成默认的 radius 




## Fourier filters - 傅里叶滤波


## Interpolation - 插值


## Measurements - 统计测量



## Morphology - 形态学变换


