- [1. 数据处理pandas包](#1-数据处理pandas包)
- [2. IO 输入输出](#2-io-输入输出)
  - [2.1. read_csv](#21-read_csv)
  - [others](#others)
  - [2.2. csv写入](#22-csv写入)
- [3. Series](#3-series)
  - [3.1. creation](#31-creation)
  - [3.2. 操纵数据](#32-操纵数据)
  - [3.3. index](#33-index)
- [4. DataFrame](#4-dataframe)
  - [4.1. 创建DF Constructor](#41-创建df-constructor)
  - [4.2. 访问数据 Indexing, iteration](#42-访问数据-indexing-iteration)
    - [4.2.1. 添加修改内容](#421-添加修改内容)
    - [4.2.2. DF切片](#422-df切片)
  - [4.3. Attributes and underlying data](#43-attributes-and-underlying-data)
    - [4.3.1. DF.info](#431-dfinfo)
    - [4.3.2. index](#432-index)
  - [4.4. 选择 Reindexing / selection / label manipulation](#44-选择-reindexing--selection--label-manipulation)
    - [4.4.1. label 操作](#441-label-操作)
      - [4.4.1.1. 根据重复内容删除行](#4411-根据重复内容删除行)
    - [4.4.2. reindexing](#442-reindexing)
      - [4.4.2.1. .reindex ()](#4421-reindex-)
  - [4.5. 计算与描述 Computations / descriptive stats](#45-计算与描述-computations--descriptive-stats)
  - [4.6. 分组以及应用函数变换 Function application, GroupBy & window](#46-分组以及应用函数变换-function-application-groupby--window)
    - [4.6.1. groupby](#461-groupby)
  - [4.7. Plotting](#47-plotting)
  - [4.8. 合并 Combining / comparing / joining / merging](#48-合并-combining--comparing--joining--merging)
  - [4.9. 缺失值处理 Missing data handling](#49-缺失值处理-missing-data-handling)
  - [4.10. 排序 sort_values sort_index](#410-排序-sort_values-sort_index)
  - [4.11. .loc 按标签提取 laber](#411-loc-按标签提取-laber)
      - [4.11.0.1. 使用下标切片](#41101-使用下标切片)
      - [4.11.0.2.](#41102)
- [5. pandas.Categorical](#5-pandascategorical)
  - [5.1. 创建 categorical](#51-创建-categorical)
- [6. pandas.Timestamp](#6-pandastimestamp)
  - [6.1. Timestamp](#61-timestamp)
  - [6.2. Timedelta](#62-timedelta)
  - [6.3. 计算](#63-计算)
  - [6.4. DatetimeIndex](#64-datetimeindex)
  - [6.5. 使用时间索引来切片](#65-使用时间索引来切片)
    - [6.5.1. 基于时间索引的统计  resample](#651-基于时间索引的统计--resample)
- [7. 处理数据](#7-处理数据)
      - [7.0.0.1. 根据缺失字段删除整行](#7001-根据缺失字段删除整行)
    - [7.0.1. sort 重新排列](#701-sort-重新排列)
  - [7.1. 统计数据](#71-统计数据)
    - [7.1.1. 基础寻值](#711-基础寻值)
    - [7.1.2. 基于数值统计](#712-基于数值统计)
      - [7.1.2.1. Series.value_counts()](#7121-seriesvalue_counts)
      - [7.1.2.2. .sum()](#7122-sum)
      - [7.1.2.3. cut与qcut](#7123-cut与qcut)
      - [7.1.2.4. crosstab](#7124-crosstab)
    - [7.1.3. 基于统计学的数值](#713-基于统计学的数值)
      - [7.1.3.1. Variance 方差](#7131-variance-方差)
# 1. 数据处理pandas包

pandas 是一种列存数据分析 API。它是用于处理和分析输入数据的强大工具，很多机器学习框架都支持将 pandas 数据结构作为输入。   
虽然全方位介绍 pandas API 会占据很长篇幅，但它的核心概念非常简单  

```py
# 导入pandas并且输出版本
import pandas as pd
pd.__version__
```

学习pandas的目标

* 大致了解 pandas 库的 DataFrame 和 Series 数据结构
* 存取和处理 DataFrame 和 Series 中的数据
* 将 CSV 数据导入 pandas 库的 DataFrame
* 对 DataFrame 重建索引来随机打乱数据

pandas 中的主要数据结构被实现为以下两类：
1. Series，它是单一列。
2. DataFrame，您可以将它想象成一个关系型数据表格，其中包含多个行和已命名的列。
3. DataFrame 中包含一个或多个 Series，每个 Series 均有一个名称。



# 2. IO 输入输出

* IO 是 Pandas 最基础的功能
  * Pandas 包中提供的读写函数都是非常高层次的函数
  * 支持了多种格式

## 2.1. read_csv

* `read_csv(filepath_or_buffer[, sep, …])`
* read_csv 是最基础的读取函数, 拥有巨多的参数
* 提供了在读取阶段就实现 Miss Value 补足的功能
* `filepath_or_buffer` 必要参数 str, path object or file-like object
  * str 可以是 URL, include http, ftp, s3, gs, and file
  * 可以接受 `os.PathLike` 的对象
  * 可以接受通过 `open()` 打开的 file object

1. 数据格式设定
   * sep: str, default `,` 逗号, 用于指定一行数据间的分隔符
   * index_col: int,str,sequence of int/str, False, default None
     * 用于指定 DataFrame 的index, 指定第几列数据是该数据集的index, 默认由 Pandas 推倒
     * 指定 False 代表该数据中没有 Index
   * header: int or list of int, default "infer"
     * 用于指定第几行数据是该 DataFrame 的列名称, 默认由 Pandas 推导
     * 如果原本数据中没有列名, 必须设置好 `header=None`
     * 如果传入了 names 参数, 则该参数自动设置成 None
   * names: array-like, default None
     * 接上个参数, 代表传入的列名, 传入的列名不能有重复
   * squeeze: boolean, default
     * 设定好该参数可以将只有一列的数据作为 Series 返回而不再是 DF

2. 特殊字符处理
   * quoting:
   * quotechar:
   * doublequote:

## others
最常用的部分
1. `read_table(filepath_or_buffer[, sep, …])`
      Read general delimited file into DataFrame.

	    Read a comma-separated values (csv) file into DataFrame.

2. `read_fwf(filepath_or_buffer[, colspecs, …])`
      Read a table of fixed-width formatted lines into DataFrame.
 
* read_table 和 read_csv 基本一样, 只不过默认的数据分隔符不相同
  * table 是 `'\t'`
  * csv 是 `','`
* read_fwf 比较少被用到  `fixed-width formatted`

csv和table的参数说明
```py
# -------------------------------------------------------
# 1. header     表示数据的列名 在哪一行 比如 默认会header=0
# 2. names      如果文件里没有列名, 则可以自己传入, 传入后会自动设定header=None
#               --这个 list 里不允许重复值
# 3. index_col  指定文件的索引,默认会使用数据中的第一列
#               --可以传入数字或者字符串， 传入列表的时候会使用多索引


# 对于没有列名的数据,自己指定列名
name = ['id', 'username', 'name','score']
cars=pd.read_csv("cars.csv",sep=';',index_col='Car',names=name)


```
## 2.2. csv写入

  data.to_csv(write_file,index=False)

# 3. Series 

* 最基础的数据类型, 和 numpy.array 有些类似  
* Series 可用作大多数 NumPy 函数的参数
* Series 虽然只有一列, 但也有 index, 默认是从0开始的整数, 也可以通过 `index=` 指定

## 3.1. creation

* 可以从许多数据类型创建 Series
* 可以指定 index, index 在pandas里可以重复, 但是有可能在别的运算中报出异常
* 可以从一个 scalar 标量中创建Series, 此时必须提供 Index, 然后该标量会被重复 index 长度次

```py
s = pd.Series(data, index=index)

# 创建一个Series
# 传入的参数可以是 python列表或者元组  numpy.array  python标量
s = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])

# index 也是一个 list of axis labels  但是一定要和数据等长
s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])

# 甚至可以从 python 字典创建
d = {"b": 1, "a": 0, "c": 2}
pd.Series(d)

# 标量创建必须给定 index
pd.Series(5.0, index=["a", "b", "c", "d", "e"])
```
## 3.2. 操纵数据

```py
import numpy as np

# pandas 
np.log(population)

# 对 Series应用python的基本运算 
population / 1000.

# 创建了一个指明 population 是否超过 100 万的新 Series
population.apply(lambda val: val > 1000000)
# 0    False
# 1     True
# 2    False
# dtype: bool

```
## 3.3. index

* 在输出 `Series`的时候会先输出输出内容,再输出内容的类型 `Name: Year, dtype: int64` , 只有单列  
* 对`Series`使用重建索引`reset_index()`会将其转换成`DataFream`并且旧的索引会变成 `Index`列

# 4. DataFrame

pandas最核心的数据类型, 由任意个 Series 和其名称组成   
可以理解为 spreadsheet 或者 SQL 表  


## 4.1. 创建DF Constructor

DataFrame accepts many different kinds of input:
* Dict of 1D ndarrays, lists, dicts, or Series
* 2-D numpy.ndarray
* Structured or record ndarray
* A Series
* Another DataFrame

创建:
1. 通过字典创建 
   1. dict: `df=pd.DataFrame( d
   2. ict() )`
   3. 通过 dict of Series  : `df=pd.DataFrame( 同样是字典dict(),value=Series  )`
      * 需要通过 字典的方式给 Series 赋予列名称
   4. dict of ndarrays/lists: 
      * `d = {"one": [1.0, 2.0, 3.0, 4.0], "two": [4.0, 3.0, 2.0, 1.0]}`
      * `pd.DataFrame(d)`

```py
# 从Series创建, 需要赋予每个Series列名称
# 如果 Series 在长度上不一致，系统会用特殊的 NA/NaN 值填充缺失的值
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
# 圆括号和大括号和冒号
pd.DataFrame({ 'City name': city_names, 'Population': population })


# 复习一下python的字典
data = {"Name"  : ["Eli", "Abe", "Ito", "Mia", "Moe", "Leo", "Mia", "Ann"],
        "Age"   : [19, 20, 21, 20, 20, 20, 20, 19],
        "Gender": ["M","M","M","F","F","M", "F","F"]
}
#字典转为 DF
df = pd.DataFrame(data)
# 直接创建
pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland.']})

# 创建随机内容 , 指定 索引和列标签
df = pd.DataFrame(np.random.randn(8, 4),
    index=['a','b','c','d','e','f','g','h'], 
    columns = ['A', 'B', 'C', 'D'])

# 手动填入数据方法, 参数依次为:
   # value的值
   # columns 的值
   # index 的样式
dfl = pd.DataFrame(np.random.randn(5, 4),columns=list('ABCD'),index=pd.date_range('20130101', periods=5))

```
## 4.2. 访问数据 Indexing, iteration


```py

# 访问数据, 可以使用 python 的 dict/list 指令访问 DataFrame数据
# 创建DF
cities = pd.DataFrame({ 'City name': city_names, 'Population': population })

# 访问某一整列 类型为 <class 'pandas.core.series.Series'>
print(type(cities['City name']))

cities['City name']
# 直接访问一整列 输出
# 0    San Francisco
# 1         San Jose
# 2       Sacramento
# Name: City name, dtype: object


# 多使用一个方括号来访问具体的数据
print(type(cities['City name'][1]))
cities['City name'][1]
# <class 'str'>
# San Jose
```
### 4.2.1. 添加修改内容

对于df类型, 直接使用标签即可添加修改内容
```py

# Converting 'name' column into uppercase, 'username' into lowercase
df['name'] = df['name'].str.upper()
df['username'] = df['username'].str.lower()

# Clearing all additional space
df.columns = df.columns.str.strip()
df['name'] = df['name'].str.strip()

# 向现有 DataFrame 用两种方法添加 Series 
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']


# 修改更改列名或者索引名
DataFrame.index = [newName]
DataFrame.columns = [newName]


# 该函数非返回值类型,而是直接生效
df.rename(columns={'原列名'：'新列名'},inplace=True) 

# 修改内容 
# Adding a column that count the length of the name
df['name_len'] = df['name'].str.len()

age=[]
for i in birthday:
  age.append(math.floor((today-i)/pd.to_timedelta(365, unit='d')))
# 直接建立新的一列名为'年龄'直接赋值
tanjoubi['年齢']=age
```

### 4.2.2. DF切片

* DataFrame是个结构化数据　`<class 'pandas.core.frame.DataFrame'>`  
* 在输出 `DataFrame`的时候会输出表头 , 再输出内容   `print(car_year)`  
* 要想对DataFrame进行切片 使用双方括号  `car_year[["Year","Day"]]`  切出来的类型也是`DataFrame`   

想要按行提取数据 , 直接使用一个方括号即可  
`homes_table[:10]`  


## 4.3. Attributes and underlying data

最基础的, DF 相关的属性
* `DataFrame.index`    The index (row labels) of the DataFrame.
* `DataFrame.columns`  The column labels of the DataFrame.
* 数据维度类 
  * `DF.size`            返回一个整数, 代表元组总数
  * `DF.shape`           返回一个元组, 代表 DF 的维度
  * `DF.ndim`            返回整数, 代表维度个数
* 数据表示
  * `DF.value`           将任意 DF 转为 numpy 表示的类型返回


### 4.3.1. DF.info

打印信息在屏幕上, 无返回值  
* Print a concise summary of a DataFrame.
* including the index dtype and columns, non-null values and memory usage.

```py
DataFrame.info(verbose=None, buf=None, max_cols=None, memory_usage=None, show_counts=None, null_counts=None)

```
### 4.3.2. index

使用`DataFrame.index`可以获取一个类数组的索引列表,依据数据的不同有多种形式    

`Int64Index([1997, 1998, 1996, 1995, 1994, 1993], dtype='int64')`  
`<class 'pandas.core.indexes.numeric.Int64Index'>`  

`RangeIndex(start=0, stop=23, step=1)`  
`<class 'pandas.core.indexes.range.RangeIndex'>`   


更改DF的索引, 将会替换原本索引,并且通过values获取数据时该列数据将不再出现  
df = df.set_index("Country")

## 4.4. 选择 Reindexing / selection / label manipulation

### 4.4.1. label 操作

* drop  : Drop specified labels from rows or columns.
* 

```py
data=data.drop(data.loc[data['State']=="CA"].index)
```

#### 4.4.1.1. 根据重复内容删除行


```py
# 在成为 DataFrame 格式后 ,可以使用 自带的方法来清除重复数据  drop_duplicates

print(df.drop_duplicates())

```
* pandas.DataFrame.drop_duplicates (Python method, in pandas.DataFrame.drop_duplicates)
* pandas.Index.drop_duplicates (Python method, in pandas.Index.drop_duplicates)
* pandas.Series.drop_duplicates (Python method, in pandas.Series.drop_duplicates)


* Parameters
  * subset : `column label `or `sequence of labels`, optional
    * Only consider` certain columns` for identifying duplicates, by `default use all of the columns.`
  * keep : {‘first’, ‘last’, False}, default ‘first’
    * Determines which duplicates (if any) to keep. 
      * first : Drop duplicates except for the first occurrence. 
      * last : Drop duplicates except for the last occurrence.
      * False : Drop all duplicates.
  * inplace : bool, default False
    * Whether to drop duplicates in place or to return a copy.
    * ignore_index : bool, default False
      * If True, the resulting axis will be `labeled 0, 1, …, n - 1. `
      * New in version 1.0.0.

### 4.4.2. reindexing

#### 4.4.2.1. .reindex ()

pandas有一个类似 .loc的方法, 为 reindex() , 会检查内容的有无并返回对应的数值  
缺点是代码长, 不简洁 ,必须要指定 `index` 和 `columns`参数  
```py
# When checking elements that possibly does not exist, use reindex()
print("\nChecking rows ['f','g','h','i'] and column ['D','E']")
print(df.reindex(index=['f','g','h','i'],columns=['D','E']))
```

## 4.5. 计算与描述 Computations / descriptive stats

可以使用多种方法来对 DF 数据进行全局描述
* 


```py

# 使用 DataFrame.describe 来显示关于 DataFrame 的有趣统计信息。
df.describe()

```
## 4.6. 分组以及应用函数变换 Function application, GroupBy & window


### 4.6.1. groupby

* 最为常用的分析工具
* 使用 groupby来分类数据 , 并使用 describe() 来生成一个默认模板的数据分析结果

```py
DataFrame.groupby(by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=<object object>, observed=False, dropna=True)
"""
as_index : 是否将分组的 by 作为 index, False 的话就是传统的 SQL groupby 返回类型

"""




# example
gender_group_desc = df.groupby("Gender").describe()
# describe()的默认模板有很多不必要的列 , 使用下面的方法来筛选只想保留的数据 
gen_gr_desc_countmean = gender_group_desc.loc[:,(slice(None),['count','mean'])]


```

## 4.7. Plotting



```py
# 借助 DataFrame.hist，可以快速了解某一个列的中值的分布： 这是pandas自带的绘图函数
df.hist('housing_median_age')
```


## 4.8. 合并 Combining / comparing / joining / merging

包含了基本的多DF连接处理, 类似于数据库 table
* append
* assign
* compare
* join
* merge
* update

```py
# 对于读取到的两个文件, 可以很方便的将他们链接, 记得要重新排列索引
h_table = h1_table.append(h2_table)
h_table = h_table.reset_index(drop=True)


# 使用concat合并几个df,axis指定各自列合并
# Concatenating all necessary data
new_df = pd.concat([id_ext, name_ext, stats_ext], axis=1)

# 使用join进行链接
df=fruit_name.join(score_mean)

# 相同列标签时, 重命名各自的列标签(在后面加入字符串)
df.join(other, lsuffix='_caller', rsuffix='_other')

# If we want to join using the key columns, we need to set key to be the index in both df and other. 
# The joined DataFrame will have key as its index.
# 如果想要以某一个 列作为 Key值进行链接 ,需要各自指定为 Index
df.set_index('key').join(other.set_index('key'))
```

## 4.9. 缺失值处理 Missing data handling


```py
# 有些时候 , 无效数据在源数据中的表示方法不一致, 可以先替换成能够识别的 np.NaN
homes_table = homes_table.replace('-',np.NaN)
# 为了防止失效数据, 可以使用 isnull() 来查找无效数据 , 无效数据可能是空白, 也可能是错误格式
print(bmi_table.isnull())
# 对于失效数据也有方法可以方便的填充它
w_sep = bmi_table[["Weight(Sep)"]]
w_sep_filled = w_sep.fillna(int(w_sep.mean()))  # 也可以使用别的值 如 .median()
# Replacing the old data with new filled one
bmi_table[["Weight(Sep)"]] = w_sep_filled  


# 在删除行后重新设定行索引
homes_table =homes_table.reset_index()
```


## 4.10. 排序 sort_values sort_index

```py

DataFrame.sort_index(ascending = False)

DataFrame.sort_values(by='mean_rating',)

```
## 4.11. .loc 按标签提取 laber 


`.loc ` 是一个很严格的类型, 如果输入的切片是整数而且不能转化成原本DF的Index,会报类型错误  
与之对应的是 `.iloc` 将会把原本的 DF 看作纯净的矩阵, 只能使用整数 `0起步` 的下标来进行切片  

```py
# index=pd.date_range('20130101', periods=5)
dfl = pd.DataFrame(np.random.randn(5, 4),columns=list('ABCD'),index=pd.date_range('20130101', periods=5))

# TypeError: cannot do slice indexing on <class 'pandas.tseries.index.DatetimeIndex'> with these indexers [2] of <type 'int'>
dfl.loc[2:3]

# success
dfl.loc['20130102':'20130104']

```


#### 4.11.0.1. 使用下标切片
```
按具体数值来切片  
df1 = pd.DataFrame(np.random.randn(6, 4),index=list('abcdef'),columns=list('ABCD'))

          A         B         C         D
a  0.132003 -0.827317 -0.076467 -1.187678
b  1.130127 -1.436737 -1.413681  1.607920
c  1.024180  0.569605  0.875906 -2.211372
d  0.974466 -2.006747 -0.410001 -0.078638
e  0.545952 -1.219217 -1.226825  0.769804
f -1.281247 -0.727707 -0.121306 -0.097883


切片的方法   .loc [  行部分 , 列部分  ]  , 

行与列部分可以分别使用 [] 来进行多选
或者使用冒号 :  单独冒号代表选择全部 , 和列表切片一模一样

df1.loc[['a', 'b', 'd'], :]
          A         B         C         D
a  0.132003 -0.827317 -0.076467 -1.187678
b  1.130127 -1.436737 -1.413681  1.607920
d  0.974466 -2.006747 -0.410001 -0.078638

同时对横纵坐标切片  
df1.loc['d':, 'A':'C']
          A         B         C
d  0.974466 -2.006747 -0.410001
e  0.545952 -1.219217 -1.226825
f -1.281247 -0.727707 -0.121306

```


#### 4.11.0.2. 
按照具体值来切
```py

# 对 a 行的所有列进行判断 , 返回一个 Series , 内容为布尔值
df.loc['a'] > 0
# 类似的 对 a b 行的所有列进行判断, 返回一个 PD ,内容为布尔值
df.loc[['a','b']]> 0

data.loc[data['State']=="CA"]
```



# 5. pandas.Categorical 

## 5.1. 创建 categorical
Categoricals can only take on only a limited, and usually fixed, number of possible values (categories).  


* Parameters：
  *   values : list-like
      * The values of the categorical. If categories are given, values not in categories will be replaced with NaN.
  *  categories: Index-like (unique), optional
     * The unique categories for this categorical. 
     * If not given, the categories are assumed to be the **unique values** of values (sorted, if possible, otherwise in the order in which they appear).
   * ordered ： bool, default False
     * Whether or not this categorical is treated as a ordered categorical. 
     * If True, the resulting categorical will be ordered. An ordered categorical respects, 
     * when sorted, the order of its categories attribute (which in turn is the categories argument, if provided).
   * dtype : CategoricalDtype
     * An instance of CategoricalDtype to use for this categorical.
     * New in version 0.21.0.
* Raises
  * ValueError
    *   If the categories do not validate.
  * TypeError
    *   If an explicit ordered=True is given but no categories and the values are not sortable.

```
>>>pd.Categorical([1, 2, 3, 1, 2, 3])
[1, 2, 3, 1, 2, 3]
Categories (3, int64): [1, 2, 3]

>>>pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c'])
[a, b, c, a, b, c]
Categories (3, object): [a, b, c]


Ordered Categoricals can be sorted according to the custom order of the categories and can have a min and max value.

>>>c = pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c'], ordered=True,categories=['c', 'b', 'a'])

[a, b, c, a, b, c]
Categories (3, object): [c < b < a]
```




# 6. pandas.Timestamp 
Both Python and pandas each have a timedelta object that is useful when doing date addition/substraction.


## 6.1. Timestamp 
```py
print(pd.to_datetime('2020-6-12'))
print(pd.to_datetime('2020 / 6 / 12'))
print(pd.to_datetime('June 12, 2020 16:20'))
print(pd.to_datetime('12 June 2020 16h20m'))
print(pd.to_datetime('12/6, 2020. 16h20m', dayfirst=True))
print(pd.to_datetime(7, unit='D', origin='JUN 5, 2020'))


# 通过设置通配符字符串  , 提取时间
time = 'Tanggal: Jun 12, 2020 Waktu: 4:20 pm'
time_format = 'Tanggal: %b %d, %Y Waktu: %I:%M %p'
print(pd.to_datetime(time, format=time_format))

# 甚至to_datetime 方法可以直接作用于 Series,将其转换成一个Timestamp的 Series
time = pd.to_datetime(date_temp["Date"], dayfirst=True,
        errors='coerce')


# 通過一個Timestamp可以方便的提取時間屬性

ts = pd.to_datetime('2020-6-12 16:20:30.3')
# [2020, 6, 12, 16, 20, 30]
ts_list1 = [ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second]

#  [4, 164, 30]
ts_list2 = [ts.dayofweek, ts.dayofyear, ts.daysinmonth]
```
## 6.2. Timedelta
represent an amount of time.


```py
#0 days 03:10:00
print(pd.to_timedelta('3 hours 10 minutes'))
#0 days 03:10:20
print(pd.to_timedelta('3:10:20'))
#0 days 03:00:00
print(pd.to_timedelta(3, unit='h')) #'h' = hours

# 转换序列
run_time = pd.read_csv("run_time_100m.csv")
time = pd.to_timedelta(run_time['Time'])


# Timedelta 同样具有一定属性
td = pd.to_timedelta('2 days 10 hours 53 minutes 30 seconds')

# Components(days=2, hours=10, minutes=53, seconds=30, milliseconds=0, microseconds=0, nanoseconds=0)
print(td.components)

# 212010.0
print(td.total_seconds())

```

## 6.3. 计算
可以方便的进行计算
```py

ts1 = pd.to_datetime('Jun 12, 2020')
ts2 = pd.to_datetime('Jul 12, 2020')
td1 = pd.to_timedelta('3 days 12 hours') 
td2 = pd.to_timedelta('30 minutes')

print(ts2 - ts1)
print(ts1 + (td1 * 2))
print(ts1 - td2)
print(td1 / td2)

```
## 6.4. DatetimeIndex
读取数据后, 可以将时间类型作为索引

```py
room_data = pd.read_csv('room_stats.csv')

# 类型转换
room_data['date'] = pd.to_datetime(room_data['date'])
# 将时间作为索引
room_data = room_data.set_index('date')

# 时间索引同样具有属性
print("Number of data in each hour of day")
# 提取所有索引的小时部分
hour_data = room_data.index.hour
# 排序并统计个数, 得出所有24中每个小时的样本数
print(hour_data.value_counts().sort_index())


# 类似的还有 weekday_name
print("\nNumber of data in each day of week")
dayofweek_data = room_data.index.weekday_name
# Monday        581
# Tuesday      1440
print(dayofweek_data.value_counts().sort_index())

# 月
print("\nNumber of data in each month of year")
month_data = room_data.index.month
print(month_data.value_counts().sort_index())
```

## 6.5. 使用时间索引来切片
```py
# 利用时间索引来进行 .loc 切片
# "Data on Feb 4, 2015"
slice1 = room_data.loc['2015-02-04'] 

# Data on February 2015"
# '2015-02' also OK
slice2 = room_data.loc['Feb 2015'] 


# Data on Feb 3, 2015 at 15 pm"
slice3 = room_data.loc['2015-02-03 15']


# Data on Feb 2, 2015 at 15:30-16:00"
slice4 = room_data.loc['2015-02-03 15:30':'2015-02-03 16:00']

```
### 6.5.1. 基于时间索引的统计  resample
```py
# The Number of Data per hour"
# 按小时分组并统计
group1 = room_data.resample('H').size()

# The Average of Temperature per day"
# 按每天分组并算平均值
group2 = room_data.resample('D').mean()
print(group2['Temp']) 

# The Maximum of Humidity and CO2 per 300 seconds"
# 按300秒来分组
group3 = room_data.resample('300S').max()
print(group3[['Humid.','CO2']])

```



    
# 7. 处理数据




#### 7.0.0.1. 根据缺失字段删除整行


* pandas.DataFrame.dropna (Python method, in pandas.DataFrame.dropna)
* pandas.Index.dropna (Python method, in pandas.Index.dropna)
* pandas.Series.dropna (Python method, in pandas.Series.dropna)

```py

# 或者直接删除无效的整行,有任何缺失字段都删除整行
table=table.dropna()
# 只删除特定字段缺失的行
table=table.dropna(subset=['List'])
# 保留 有效字段在 thresh 个及以上的行
table=table.dropna(thresh=7)

```



### 7.0.1. sort 重新排列

* Index
  * pandas.DataFrame.sort_index (Python method, in pandas.DataFrame.sort_index)
  * pandas.Series.sort_index (Python method, in pandas.Series.sort_index)
* Values
  * pandas.DataFrame.sort_values (Python method, in pandas.DataFrame.sort_values)
  * pandas.Index.sort_values (Python method, in pandas.Index.sort_values)
  * pandas.Series.sort_values (Python method, in pandas.Series.sort_values)

按照DataFrame的参数  
* value独有的参数
  * by ： str or list of str
    * Name or list of names to sort by.
    * Allow specifying index or column level names.
      * if axis is 0 or `‘index’` then by 
        * may contain index levels and/or column labels.
      * if axis is 1 or `‘columns’ `then by may contain column levels and/or index labels.
  * axis : {0 or ‘index’, 1 or ‘columns’}, default 0
    * Axis to be sorted.
    * ascendingbool or list of bool, default True
    * Sort ascending vs. descending. Specify list for multiple sort orders. If this is a list of bools, must match the length of the by.
  * inplace : bool, default False
    * If True, perform operation in-place.
    * Returns
      * sorted_objDataFrame or None
      * DataFrame with sorted values if inplace=False, None otherwise.
  * kind : {‘quicksort’, ‘mergesort’, ‘heapsort’}, default ‘quicksort’
    * Choice of sorting algorithm. 
    * See also ndarray.np.sort for more information. 
    * mergesort is the only stable algorithm. 
    * For DataFrames, this option is only applied when sorting on a `single column or label`.
  * na_position : {‘first’, ‘last’}, default ‘last’
    * Puts NaNs at the beginning if first; last puts NaNs at the end.
  * ignore_index : bool, default False
    * If True, the resulting axis will be labeled 0, 1, …, n - 1.







## 7.1. 统计数据

使用统计方法将会使得数据对象降维  DF->Series   Series->numpy

### 7.1.1. 基础寻值
* 平均 .mean()
* 标准差 .std()
* 最大最小值中值 .max() .min() .median()
* 按在范围内的百分比获取值  .quantile()
  * quantile([0, .25, .50, .75, 1])
  * 分别等于  .min()  第四分之一  .median() 第四分之三  .max()
```py

# 获取平均值
player_table.mean() 

# numeric_only=True 只考虑数字部分的value 
# 获取中值
player_table.median(numeric_only=True)

# 标准差
player_table.std(numeric_only=True)

# 通过最大最小值获取范围, 相同长度和维度的数据可以相减  
print("\nRange")
print(player_table.max(numeric_only=True) -player_table.min(numeric_only=True))
```

### 7.1.2. 基于数值统计

#### 7.1.2.1. Series.value_counts()
```py
# 对Series使用value_count来进行数据统计, 统计的按照相同数据来进行统计  
# 得到的返回值仍是一个 Series
car_table['Year'].value_counts() 
```

#### 7.1.2.2. .sum()

```py
# 对一个DataFrame使用 得到的结果是一个 Series
# 原本的列标签变为索引 `<class 'pandas.core.indexes.base.Index'>`
hurr_table_sum = pd.DataFrame(hurr_table.sum())



# 将数据变为百分比
s_percentage = (s/sum(s)*100).round(2)
```

#### 7.1.2.3. cut与qcut

分组数据  

* pandas.cut(x, bins, right: bool = True, labels=None, retbins: bool = False, precision: int = 3, include_lowest: bool = False, duplicates: str = 'raise')
  * Bin values into discrete intervals.
* pandas.qcut(x, q, labels=None, retbins: bool = False, precision: int = 3, duplicates: str = 'raise')  
  * Quantile-based discretization function.

#### 7.1.2.4. crosstab
分析数据关联性  
By matching different categorical frequency distributions,   
you can display the relationship between qualitative variables.    
The pandas.crosstab function can match variables or groups of variables,   
helping to locate possible data structures or relationships.    

```py

# 获取身高并按组分类  
binned_ages = pd.qcut(player_table["Height(inches)"], q=5, precision=0)

# 获取crosstab
position_vs_ages = pd.crosstab(player_table["Position"],binned_ages)


# We can add an extra argument normalize, to see the proportion for each part. 
position_vs_ages = pd.crosstab(player_table["Position"],binned_ages,normalize='columns')*100


```




### 7.1.3. 基于统计学的数值



* 偏度（skewness）
  * 是统计数据分布偏斜方向和程度的度量，是统计数据分布非对称程度的数字特征。  
  * 偏度定义中包括正态分布（偏度=0），右偏分布（也叫正偏分布，其偏度>0），左偏分布（也叫负偏分布，其偏度<0）
* 峰度（peakedness；kurtosis）又称峰态系数
  * 表征概率密度分布曲线在平均值处峰值高低的特征数。
  * 直观看来，峰度反映了峰部的尖度。随机变量的峰度计算方法为：随机变量的四阶中心矩与方差平方的比值。
  *  If the kurtosis is greater than 3, then the dataset has heavier tails than a normal distribution (more in the tails).
  *  If the kurtosis is less than 3, then the dataset has lighter tails than a normal distribution (less in the tails). 
*  协方差 correlation
  

nonparametric correlation, such as a Spearman  


The chi-square statistic tells you when the table distribution of two variables is statistically comparable to a table 
  in which the two variables are hypothesized as not related to each other (the so-called independence hypothesis).

```py
# 以series为对象
series.skew()
series.kurt()

# correlation 协方差
table.cov()
table.corr()



```
#### 7.1.3.1. Variance 方差

可以通过Series获取方差  

```py

# 筛选数值  
group1 = player_table["Position"] == 'Catcher'
group2 = player_table["Position"] == 'Outfielder'

# 获取 Series
variable = player_table["Height(inches)"]

# 计算方差  
variance1 = variable[group1].var().round(2)
variance2 = variable[group2].var().round(2)

print(f"'Catcher' Variance: {variance1}")
print(f"'Outfielder' Variance: {variance2}")

```

