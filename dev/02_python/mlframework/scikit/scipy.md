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

# 5. Signal Processing (scipy.signal)

SciPy 中专门用于处理信号的包
* 包括了一些 filter
* 一些 filter design tools
* 一些 B-spline interpolation algorithms for 1- and 2-D data

## 5.1. Filtering

Filtering : a generic name for any system that modifies an input signal in some way.

在 SciPy 中, signal 也是 Numpy array  


## 5.2. B-splines


# 6. Statistics ( scipy.stats ) 
 
### 6.0.1. 偏度（skewness） 和 峰度（peakedness；kurtosis）又称峰态系数
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


###  6.0.2. T-Test

```py

from scipy.stats import ttest_ind

group1 = player_table["Position"] == 'Catcher'
group2 = player_table["Position"] == 'Outfielder'
variable = player_table["Height(inches)"]

t, pvalue = ttest_ind(variable[group1], variable[group2],axis=0, equal_var=False)

print(f"t statistic {t}")
print(f"p-value {pvalue}")

```

### 6.0.3. ANOVA

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

### 6.0.4. from  chi-square statistic

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

### 6.0.5. pearson
a = player_table["Height(inches)"]
b = player_table["Weight(lbs)"]
#print(a)
rho_coef, rho_p = spearmanr(a, b)
r_coef, r_p = pearsonr(a, b)

print(f'Pearson r: {r_coef}')
print(f'Spearman r: {rho_coef}')

# 7. Multidimensional image processing ( scipy.ndimage )