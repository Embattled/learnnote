# 1. File System

os, glob,  shutil
## pathlib
The pathlib module was introduced in Python 3.4 to deal with these challenges. It gathers the necessary functionality in one place and makes it available through methods and properties on an easy-to-use Path object.  

### pathlib.Path
The best way to construct a path is to join the parts of the path using the special operator `/`.


#### 使用Path来定义路径
You can use `Path.cwd()` or `Path('.') `to refer to your currently working directory.

```py
from pathlib import Path

print("Getting 'text1.txt'")
path1 = Path('.') / 'folder1' / 'text1.txt'
print(path1)

```

#### 使用Path来获取路径的属性

.name, .parent, .stem, .suffix, .anchor 

The pathlib.Path is represented by either a `WindowsPath` or a `PosixPath`.

```py
path1 = Path('.') / 'folder1' / 'text1.txt'
print([path1, path1.name, path1.stem, path1.suffix, path1.parent, path1.parent.parent, path1.anchor])
# [PosixPath('folder1/text1.txt'), 'text1.txt', 'text1', '.txt', PosixPath('folder1'), PosixPath('.'), '']

```

#### 获取文件列表

Using `.iterdir()` you can get all the files in a folder.   
By list comprehension, you can convert this into a list object.  


```py
path2 = Path('.') / 'folder1'
path_list = list(path2.iterdir())
print(f'List of files: {path_list}')
'''
List of files: [PosixPath('folder1/text1.txt'), PosixPath('folder1/text2.txt'), PosixPath('folder1/text3.txt')]
'''

print(f'Number of files: {len(path_list)}')

```

#### Path 的有用方法

```py

# Path.exists()
# Checks if a path exists or not. Returns boolean value.
file_path = Path('.') / 'folder1' / 'text2.txt'
print(file_path.exists())


# Path.glob()   Globs and yields all file paths matching a specific pattern. 
#  mark (?), which stands for one character.
print("\nGetting all files with .csv extension.")
dir_path = Path('.') / 'folder1'
file_paths = dir_path.glob("*.csv")
print(list(file_paths))

# Path.rglob()
# This is like Path.glob method but matches the file pattern recursively.
print("\nGetting all .txt files starts with 'other' in all directories.")
dir_path = Path('.')
file_paths = dir_path.rglob("other?.txt")
print(list(file_paths))

# Path.mkdir()
# Creates a new directory at this given path. 
dir_path = Path('.') / 'folder_new' / 'folder_new_1'
# parents:(boolean) If parents is True, 
#         any missing parents of this path are created as needed. 
#          Otherwise, if the parent is absent, FileNotFoundError is raised.
dir_path.mkdir(parents=True)
# exist_ok: (boolean) 
#       If False, FileExistsError is raised if the target directory already exists. 
#       If True, FileExistsError is ignored.

 
# Path.rename(target)   This will raise FileNotFoundError if the file is not found
dir_path = Path('.') / 'folder_new' / 'folder_new_1'
dir_path.rename(dir_path.parent / 'folder_n1')


# Replaces a file or directory to the given target. Returns the new path instance. 
dir_path = Path('.') / 'folder_new' / 'folder_n1'
dir_path2 = Path('.') / 'folder1'  
dir_path.replace(dir_path.parent / dir_path2)



# Path.rmdir()
# Removes a path pointing to a file or directory. The directory must be empty, otherwise, OSError is raised.
```


# 2. numpy python数值包

## 2.1. 

### 2.1.1. np.set_printoptions

取消科学计数法显示数据  
np.set_printoptions(suppress=True)  


### 2.1.2. numpy.array

多重方括号`[]`,可以代表矩阵  

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
### 2.1.3. numpy.shape

获取目标的维度

```py

# 计算一个矩阵的奇异值
A = np.array([[1,3,4], [2,3,5], [1,2,3], [5,4,6]])
U, s, Vh = np.linalg.svd(A, full_matrices=False)

print(np.shape(U), np.shape(s), np.shape(Vh))

'''输出
(4, 3) (3,) (3, 3)
'''
```

### 2.1.4. numpy.dot()  矩阵点乘

np.diag(s)  将数组变成对角矩阵  

使用numpy进行矩阵乘法  

```py
# 使用svp分解矩阵
U, s, Vh = np.linalg.svd(A, full_matrices=False)
#使用 .dot() 将原本的矩阵乘回来 
Us = np.dot(U, np.diag(s))
UsVh = np.dot(Us, Vh)

```


## 2.2. linalg 

### 2.2.1. SVD 奇异值分解

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

# 3. 数据处理pandas包


## 3.1. DataFrame 与 Series 类型与切片

### 3.1.1. 创建DF

```py
import pandas as pd

# 直接读取 , 设定索引  , 设定文件中的分隔符
cars=pd.read_csv("cars.csv",sep=';',index_col='Car')


# 字典
data = {"Name"  : ["Eli", "Abe", "Ito", "Mia", "Moe", "Leo", "Mia", "Ann"],
        "Age"   : [19, 20, 21, 20, 20, 20, 20, 19],
        "Gender": ["M","M","M","F","F","M", "F","F"]
}

# 创建随机内容 , 指定 索引和列标签
df = pd.DataFrame(np.random.randn(8, 4),
    index=['a','b','c','d','e','f','g','h'], 
    columns = ['A', 'B', 'C', 'D'])



#字典转为 DF
df = pd.DataFrame(data)

# 手动填入数据
# value的值
# columns 的值
# index 的样式
dfl = pd.DataFrame(np.random.randn(5, 4),columns=list('ABCD'),index=pd.date_range('20130101', periods=5))



```
#### 3.1.1.1. DF切片
DataFrame是个结构化数据　`<class 'pandas.core.frame.DataFrame'>`  
在输出 `DataFrame`的时候会输出表头 , 再输出内容  
`print(car_year)`  
要想对DataFrame进行切片 使用双方括号  `car_year[["Year","Day"]]`  切出来的类型也是`DataFrame`   

想要按行提取数据 , 直接使用切片即可  
`homes_table[:10]`  

### 3.1.2. 读取

```py
# read_table 将文件中的数据以一个表排列的字符串读入
animal_table = pd.read_table("animal.txt")


# A CSV file provides more formatting than a simple text file. 
# Its header defines each of the fields. The entries are usually separated by comma.
city_table = pd.io.parsers.read_csv("cities.csv")

# 指定源文件中数据的分隔符来正确的读取
city_table = pd.read_csv("cities.csv",sep=';')
# 此时的数据属于 DataFrame  的格式 , 可以通过字段名提取整列数据
cities_data = city_table['City'] 



# 对于没有列名的数据,需要自己指定列名
header = ['id', 'username', 'name','score']
# 提前指定好列数据的类型, 可以在之后方便处理
dtype_dic = {'id': object, 'username': object,'name': object, 'score': float}
# 做好准备后再读取
df = pd.read_csv('data.csv', names=header, dtype=dtype_dic)

```

### 3.1.3. 添加修改内容
对于df类型, 直接使用标签即可添加修改内容
```py
# Converting 'name' column into uppercase, 'username' into lowercase
df['name'] = df['name'].str.upper()
df['username'] = df['username'].str.lower()

# Clearing all additional space
df.columns = df.columns.str.strip()
df['name'] = df['name'].str.strip()



# 修改更改列名或者索引名
# 直接使用 使用DataFrame.index = [newName]，DataFrame.columns = [newName]，这两种方法可以轻松实现。

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
### 3.1.4. 合并 join  concat  append
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
### 3.1.5. Series单列
注意其与`<class 'pandas.core.series.Series'>` 的区别 , DataFrame使用单方框切片的时候为Series类型,`Series`也有索引    
`print(car_year["Year"])`  只能截取单列    
在输出 `Series`的时候会先输出输出内容  , 再输出内容的类型 `Name: Year, dtype: int64` , 只有单列  
对`Series`使用重建索引`reset_index()`会将其转换成`DataFream`并且旧的索引会变成 `Index`列


### 3.1.6. values
使用`DataFrame.values` 获取到的类型为 `<class 'numpy.ndarray'>`, 为双方括号形式, 没有索引 , `[ [第一整行内容] [第二整行内容]... ]  `  
再使用`DataFrame.values.flatten()`  类型仍为`<class 'numpy.ndarray'>` , 但是变为单方括号 `[  第一整行内容  第二整行内容 ... ]`
`.flatten()` 是 `ndarray`的方法  
使用`Series.values` 获取到的类型同样为 `<class 'numpy.ndarray'>`, 直接就是单方括号形式, `[ 第一个值  第二个值 ... ]  `  

### 3.1.7. index
使用`DataFrame.index`可以获取一个类数组的索引列表,依据数据的不同有多种形式    

`Int64Index([1997, 1998, 1996, 1995, 1994, 1993], dtype='int64')`  
`<class 'pandas.core.indexes.numeric.Int64Index'>`  

`RangeIndex(start=0, stop=23, step=1)`  
`<class 'pandas.core.indexes.range.RangeIndex'>`   


更改DF的索引, 将会替换原本索引,并且通过values获取数据时该列数据将不再出现  
df = df.set_index("Country")

### 排序 sort_values sort_index

```py

DataFrame.sort_index(ascending = False)

DataFrame.sort_values(by='mean_rating',)

```
### 3.1.8. .loc 按标签提取 laber 


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


#### 3.1.8.1. 使用下标切片
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


#### 3.1.8.2. 
按照具体值来切
```py

# 对 a 行的所有列进行判断 , 返回一个 Series , 内容为布尔值
df.loc['a'] > 0
# 类似的 对 a b 行的所有列进行判断, 返回一个 PD ,内容为布尔值
df.loc[['a','b']]> 0

data.loc[data['State']=="CA"]
```

#### 3.1.8.3. .reindex ()

pandas有一个类似 .loc的方法, 为 reindex() , 会检查内容的有无并返回对应的数值  
缺点是代码长, 不简洁 ,必须要指定 `index` 和 `columns`参数  
```py
# When checking elements that possibly does not exist, use reindex()
print("\nChecking rows ['f','g','h','i'] and column ['D','E']")
print(df.reindex(index=['f','g','h','i'],columns=['D','E']))
```

## 3.2. pandas.Categorical 

### 3.2.1. 创建 categorical
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




## 3.3. pandas.Timestamp 
Both Python and pandas each have a timedelta object that is useful when doing date addition/substraction.


### 3.3.1. Timestamp 
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
### 3.3.2. Timedelta
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

### 3.3.3. 计算
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
### 3.3.4. DatetimeIndex
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

### 3.3.5. 使用时间索引来切片
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
#### 3.3.5.1. 基于时间索引的统计  resample
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

## 3.4. 写


  data.to_csv(write_file,index=False)

    
## 3.5. 处理数据

###  3.5.1. Drop数据

```py
data=data.drop(data.loc[data['State']=="CA"].index)


```


#### 3.5.1.1. 根据重复内容删除行


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


#### 3.5.1.2. 根据缺失字段删除整行


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



### 3.5.2. sort 重新排列

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






### 3.5.3. groupby
```py

# 可以使用 groupby来分类数据 , 并使用 describe() 来生成一个默认模板的数据分析结果
gender_group_desc = df.groupby("Gender").describe()

# describe()的默认模板有很多不必要的列 , 使用下面的方法来筛选只想保留的数据 
gen_gr_desc_countmean = gender_group_desc.loc[:,(slice(None),['count','mean'])]



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
## 3.6. 统计数据

使用统计方法将会使得数据对象降维  DF->Series   Series->numpy

### 3.6.1. 基础寻值
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

### 3.6.2. 基于数值统计

#### 3.6.2.1. Series.value_counts()
```py
# 对Series使用value_count来进行数据统计, 统计的按照相同数据来进行统计  
# 得到的返回值仍是一个 Series
car_table['Year'].value_counts() 
```

#### 3.6.2.2. .sum()

```py
# 对一个DataFrame使用 得到的结果是一个 Series
# 原本的列标签变为索引 `<class 'pandas.core.indexes.base.Index'>`
hurr_table_sum = pd.DataFrame(hurr_table.sum())



# 将数据变为百分比
s_percentage = (s/sum(s)*100).round(2)
```

#### 3.6.2.3. cut与qcut

分组数据  

* pandas.cut(x, bins, right: bool = True, labels=None, retbins: bool = False, precision: int = 3, include_lowest: bool = False, duplicates: str = 'raise')
  * Bin values into discrete intervals.
* pandas.qcut(x, q, labels=None, retbins: bool = False, precision: int = 3, duplicates: str = 'raise')  
  * Quantile-based discretization function.

#### 3.6.2.4. crosstab
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




### 3.6.3. 基于统计学的数值



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
#### 3.6.3.1. Variance 方差

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


# 4. matplotlib包 图表制作
Plot show graphically what you've defined numerically

导入包  
`import matplotlib.pyplot as plt`  

---
## 4.1. 折线图 plot

### 4.1.1. 画图基本
```py
# 画图库常与pandas一起用 , csv文件  
hurricane_table = pd.read_csv("hurricane.csv")

# 设置坐标轴标签
plt.xlabel('Months')
plt.ylabel('Number of Hurricanes')

# plt.plot( x轴, y轴)
# 注意后一个参数的格式 
plt.plot(range(5,13),hurricane_table[["2005"]].values)

# 可以运行多次 就可以在同一个图中画多条数据
plt.plot(range(5,13),hurr_table[["2015"]].values)

plt.show()

```



### 4.1.2. 折线图线的颜色及样式 标识 参数 

对于网格和折线都可以对其进行样式设置
```py
# 设置网格的线的类型
ax.grid(ls = ':')
# 画线的
plt.plot(range(5,13),hurr_table[["2005"]].values,ls= '-')
```
**线的类型**  
* '-'   : Solid Line
* '--'  : Dashed Line
* '-.'  : Dash-dot Line
* ':'   : Dotted Line

**线的颜色**
```py
plt.plot(range(5,13),hurr_table[["2005"]].values, ls='-', color='b', marker='s')
plt.plot(range(5,13),hurr_table[["2015"]].values, ls='--', color='c', marker='o')
```

[线的颜色文档](https://matplotlib.org/3.1.0/api/colors_api.html#module-matplotlib.colors)  
* 'b'	 Blue
* 'g'	 Green
* 'r'	 Red
* 'c'  Cyan
* 'm'  Magenta
* 'k'  Black
* 'w'  White

Markers add a special symbol to each data point in a line graph.   
[数据点的样式](https://matplotlib.org/api/markers_api.html)

* '.'   Point
* ','   Pixel
* 'o'   Circle
* 's'   Square
* '*'   Star
* '+'   Plus
* 'x'   X
```py
plt.plot(range(1,13),aritravel[["1958"]].values, ls='-',color='r',marker='s')
plt.plot(range(1,13),aritravel[["1960"]].values, ls='--',color='b',marker='o')
```

## 4.2. 饼图 pie


`matplotlib.pyplot.pie(values, colors=colors, labels=labels,explode=explode, autopct='%1.1f%%', shadow=True)` 

* x : array-like 第一个参数, 权重数列 , 会自动根据 x/sum(x) 为每个部位分配大小
* explode : array-like default: None  突出某块部分 , 数组内容是对应每块的偏移位移
* labels : list , 对每块的标签
* colors : array-like, optional, default: None . A sequence of matplotlib color args. 对每块进行指定上色
* autopct : None (default), str, or function, optional .饼块内的标签
* shadow : bool, optional, default: False . Draw a shadow beneath the pie.





## 4.3. bar  条形图
创建一个条形图 x坐标和其对应的值是必须的  

```py

# 使用一个range来代表有多少条数据 , 作为x轴的坐标 , 这并不是最终显示在图上的x轴 , 只是作为索引
#  方便后面的参数赋值
x_coord = range(0,len(hurr_table_sum))
colors = ['g' for i in x_coord]
widths = [0.7 for i in x_coord]

# 使用一个类数组来代表 scalars
values = hurr_table.sum().values

# 画图
plt.bar(x_coord,values,color=colors, width=widths)
```

有两个参数是必须的 , x 和 height
matplotlib.pyplot.bar(x,height)  
* x : sequence of scalars
* height : scalar or sequence of scalars 
* width : scalar or array-like, optional    The width(s) of the bars (default: 0.8).
* bottom : scalar or array-like, optional   The y coordinate(s) of the bars bases (default: 0).
* align : {'center', 'edge'}, optional, default: 'center'  代表条状图条的位置在哪 , 默认条的中心在坐标点上 ,edge代表 条的左边缘在坐标点上
* tick_label : str or array-like, optional The tick labels of the bars. Default: None (Use default numeric labels.)
* linewidth : scalar or array-like, optional  Width of the `bar edge(s)`. If 0, don't draw edges.



## 4.4. 直方图 Histograms  matplotlib.pyplot.hist

### 4.4.1. 单数据图
* 直方图是一种统计报告图，形式上也是一个个的长条形，  
* 但是直方图用长条形的面积表示频数，所以长条形的高度表示频数/组距，宽度表示组距，其长度和宽度均有意义。  
* 当宽度相同时，一般就用长条形长度表示频数。  

```py
plt.hist(values, bins=17, histtype='stepfilled',align='mid', color='g')
plt.hist(data, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
plt.hist(flights['arr_delay'], color = 'blue', edgecolor = 'black',bins = int(180/5))
# 绘制直方图
# data:必选参数，绘图数据
# bins:直方图的长条形数目，可选项，默认为10
# normed:是否将得到的直方图向量归一化，可选项，默认为0，代表不归一化，显示频数。normed=1，表示归一化，显示频率。
# facecolor:长条形的颜色
# edgecolor:长条形边框的颜色
# alpha:透明度

```
参数  
* x : 唯一的必要参数 (n,) array or sequence of (n,) arrays
* bins :  int or sequence or str, optional .
  * bins 就是组距
  * If bins is an integer, it defines the number of equal-width bins in the range.
  * If bins is a sequence, it defines the bin edges, including the left edge of the first bin and the right edge of the last bin
  * If bins is a string, it is one of the binning strategies supported by numpy.histogram_bin_edges:
    *  'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.
* range : tuple or None, optional  Default is None
  * If not provided, range is (x.min(), x.max()). 
  * Range has no effect if bins is a sequence.
  * The lower and upper range of the bins. Lower and upper outliers are ignored. 
* align : {'left', 'mid', 'right'}, optional
  * Controls how the histogram is plotted.
    * 'left': bars are centered on the left bin edges.
    * 'mid': bars are centered between the bin edges.
    * 'right': bars are centered on the right bin edges.



  * The type of histogram to draw. 'bar' is a traditional bar-type histogram.
  * 
### 4.4.2. 多数据图  



if we pass in a list of lists, matplotlib will put the bars side-by-side.　　
如果我们给matplotlib传递一个列表的列表, 将会自动以side-by-side的形式生成图  
通过指定参数还可以以 stack格式生成  

when we want to compare the distributions of one variable across multiple categories  

```py
# Make a separate list for each airline
x1 = list(flights[flights['name'] == 'United Air Lines Inc.']['arr_delay'])
x2 = list(flights[flights['name'] == 'JetBlue Airways']['arr_delay'])
x3 = list(flights[flights['name'] == 'ExpressJet Airlines Inc.']['arr_delay'])
x4 = list(flights[flights['name'] == 'Delta Air Lines Inc.']['arr_delay'])
x5 = list(flights[flights['name'] == 'American Airlines Inc.']['arr_delay'])

colors = ['#E69F00', '#56B4E9', '#F0E442', '#009E73', '#D55E00']
names = ['United Air Lines Inc.', 'JetBlue Airways', 'ExpressJet Airlines Inc.','Delta Air Lines Inc.', 'American Airlines Inc.']

# Make the histogram using a list of lists  将会生成side-by-side格式
# Normalize the flights and assign colors and names 
plt.hist([x1, x2, x3, x4, x5], bins = int(180/15), normed=True,color = colors, label=names)
         
# Stacked histogram with multiple airlines
plt.hist([x1, x2, x3, x4, x5], bins = int(180/15), stacked=True,normed=True, color = colors, label=names)




#　为每个数据添加注解　, 需要在画图的时候指定 'label'属性
plt.legend()

```
side-by-side 缺点  
* the bars don’t align with the labels
* still hard to compare distributions between airlines.


## 4.5. 箱线图  plt.boxplot

* 箱线图能够明确的展示离群点的信息，同时能够让我们了解数据是否对称，数据如何分组、数据的峰度.  

箱线图是一种基于五位数摘要（“最小”，第一四分位数（Q1），中位数，第三四分位数（Q3）和“最大”）显示数据分布的标准化方法
* 最小数（不是“最小值”）和数据集的中位数之间的中间数；
* 第一个四分位数（Q1 / 25百分位数）：
* 中位数（Q2 / 50th百分位数）：数据集的中间值；
* 第三四分位数（Q3 / 75th Percentile）：数据集的中位数和最大值之间的中间值（不是“最大值”）；
* 四分位间距（IQR）：第25至第75个百分点的距离；
* 晶须（蓝色显示）离群值（显示为绿色圆圈）
* IQR = Q3-Q1
  * “最大”：Q3 + 1.5 * IQR
  * “最低”：Q1 -1.5 * IQR

尽管与直方图或密度图相比，箱线图似乎是原始的  
但它们具有占用较少空间的优势，这在比较许多组或数据集之间的分布时非常有用。  


使用 pyplot可以创建箱图, 同时 pandas里也有boxplot函数  

```py

plt.boxplot(values, sym='rx', widths=.75)

#  同时创建好多箱图  根据'position' 来分组
boxplots = player_table.boxplot(column='Height(inches)', by='Position', fontsize=6, rot=90)
plt.show()




```
* sym : str, optional   设置异常点的格式 , 例如 `'b+'` 蓝色加号  `'rx'` 红色 叉号
  * The default symbol for flier points
  * Enter an empty string ('') if you don't want to show fliers. 
  * If None, then the fliers default to `'b+' `If you want more control use the flierprops kwarg.
* width : sscalar or array-like
  * Sets the width of each box either with a scalar or a sequence. 
  * The default is 0.5, or 0.15*(distance between extreme positions), if that is smaller.

## 4.6. 散点图 Scatterplots

Scatterplots show clusters of data rather than trends (as with line graphs) or discrete values (as with bar charts).  
The purpose of a scatterplot is to help you see data patterns.  

```py

homes_table = pd.io.parsers.read_csv("homes.csv")

x = homes_table[["Sell"]].values
y = homes_table[["Taxes"]].values

plt.scatter(x, y, s=[100], marker='x', c='r')

```

### 4.6.1. 使用numpy 来分析散点数据 


```py
import numpy as np
import matplotlib.pylab as plb

z = np.polyfit(x.flatten(), y.flatten(), 1)
p = np.poly1d(z)

# 注意 y轴 是 p(x)
plb.plot(x, p(x), 'k-')

```


## 4.7. 图的设置

### 4.7.1. matplotlib.pyplot.subplots 子图

分割图形 , 画多个图  
返回值会自动得到用于设置参数的 `axes.Axes` 对象

```py
# axaxes.Axes object or array of Axes objects.

# using the variable ax for single a Axes
fig, ax = plt.subplots()

# using the variable axs for multiple Axes
fig, axs = plt.subplots(2, 2)

# using tuple unpacking for multiple Axes
fig, (ax1, ax2) = plt.subplot(1, 2)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplot(2, 2)
```

### 4.7.2. 坐标轴设置 axes
```py
# 获得一个axes变量 , 然后就可以通过这个变量来对图标的坐标轴进行设置
ax = plt.axes()
# 开启网格
ax.grid()
# 可以设置网格线 , 线的格式以及颜色
ax.grid(ls = ':')
ax.grid(ls = '--',color='w')
# 设置 x y 轴的范围
ax.set(ylim=(4000 , 6000))
# 设置整个坐标轴的背景颜色
ax.set_facecolor(color='tab:gray')
```

### 4.7.3. 标签设置 坐标轴标签  图的标签 
```py
# 设置坐标轴标签 , 指明x y轴分别是什么
plt.xlabel('Months')
plt.ylabel('Number of Hurricanes')
# 设置图的标题
plt.title("MLB Players Height")

####  matplotlib.pyplot.xticks(ticks=None, labels=None)  

# 通过 .xticks 可以设置x轴上具体的坐标内容 , (索引,链接的坐标)
# 设置完成后就把具体的内容链接到了数字的索引上
plt.xticks(range(5,13),hurr_table[["Month"]].values.flatten())
# 同理 Plt.yticks() 也一样

# Pass no arguments to return the current values without modifying them.
# 获取当前坐标轴标签
label = plt.xticks()

# Passing an empty list removes all xticks.
plt.xticks([])
```
###  4.7.4. 为折线图的线添加注解 

```py
# 两条折线
plt.plot(range(1,13),aritravel[["1958"]].values, ls='-',color='r',marker='s')
plt.plot(range(1,13),aritravel[["1960"]].values, ls='--',color='b',marker='o')

# 创建一个解释 , 指明每条折线分别是什么 , 似乎是按照折线添加的顺序自动链接
# facecolor = None 代表透明
plt.legend(['1958', '1960'], loc=2,facecolor ="None")
# loc 是位置 , 1234分别是从左上角开始的逆时针旋转

```

## 4.8. Annotations

###　matplotlib.pyplot.annotate

`matplotlib.pyplot.annotate(s, xy, *args, **kwargs)`  
* xy是最基础的坐标,必须参数    
* text : str  , 标注的文字 , 必须参数
* xytext : (float, float), optional  文字的坐标 , 默认就在xy 上 ,若指定了文字坐标, 就可以添加 text 指向 xy 的箭头
* arrowprops :  dict  参数是个字典
  * `arrowprops=dict(arrowstyle="->",connectionstyle="arc3`
  * 分为两种 如果字典中没有指明参数`'arrowstyle'` 则是手动模式
    * width 	The width of the arrow in points
    * headwidth 	The width of the base of the arrow head in points
    * headlength 	The length of the arrow head in points
    * shrink 	Fraction of total length to shrink from both ends
    * ? 	Any key to matplotlib.patches.FancyArrowPatch
  * 如果指定了`'arrowstyle'` , 则可以使用一些默认设定
    * '-' 	None
    * '->' 	head_length=0.4,head_width=0.2
    * '-[' 	widthB=1.0,lengthB=0.2,angleB=None
    * '|-|' 	widthA=1.0,widthB=1.0
    * 'fancy' 	head_length=0.4,head_width=0.4,tail_width=0.4
    * 'simple' 	head_length=0.5,head_width=0.5,tail_width=0.2
    * 'wedge' 	tail_width=0.3,shrink_factor=0.5



## 4.9. Text properties

对于一切字符实例,都可以指定其属性  (e.g., title(), xlabel() and text()).
The matplotlib.text.Text instances have a variety of properties which can be configured via keyword arguments to the text commands   

https://matplotlib.org/tutorials/text/text_props.html#sphx-glr-tutorials-text-text-props-py



# 5. seaborn 数据可视化包

## 5.1. density plot AND Histogram 


A density plot is a smoothed, continuous version of a histogram estimated from the data.   


```py

sns.distplot(flights['arr_delay'], hist=True, kde=False, bins=int(180/5), color = 'blue',hist_kws={'edgecolor':'black'})
sns.distplot(flights['arr_delay'], hist=True, kde=True, bins=int(180/5), color = 'darkblue', hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 4})

```

## 5.2. swarmplot 

Draw a categorical scatterplot with non-overlapping points.  
This function is similar to stripplot(), but the points are adjusted (only along the categorical axis) so that they don’t overlap.   
This gives a better representation of the `distribution of values` , but it does `not scale well to large numbers of observations` .   
This style of plot is sometimes called a “beeswarm”.  


# 6. scipy

Scipy是一个用于数学、科学、工程领域的常用软件包，可以处理插值、积分、优化、图像处理、常微分方程数值解的求解、信号处理等问题。  
它用于有效计算Numpy矩阵，使Numpy和Scipy协同工作，高效解决问题。  

| 模块名            | 应用领域           |
| ----------------- | ------------------ |
| scipy.cluster     | 向量计算/Kmeans    |
| scipy.constants   | 物理和数学常量     |
| scipy.fftpack     | 傅立叶变换         |
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


## 6.1. scipy.stats  统计包
 
### 6.1.1. 偏度（skewness） 和 峰度（peakedness；kurtosis）又称峰态系数
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


###  6.1.2. T-Test

```py

from scipy.stats import ttest_ind

group1 = player_table["Position"] == 'Catcher'
group2 = player_table["Position"] == 'Outfielder'
variable = player_table["Height(inches)"]

t, pvalue = ttest_ind(variable[group1], variable[group2],axis=0, equal_var=False)

print(f"t statistic {t}")
print(f"p-value {pvalue}")

```

### 6.1.3. ANOVA

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

### 6.1.4. from  chi-square statistic

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

### 6.1.5. pearson
a = player_table["Height(inches)"]
b = player_table["Weight(lbs)"]
#print(a)
rho_coef, rho_p = spearmanr(a, b)
r_coef, r_p = pearsonr(a, b)

print(f'Pearson r: {r_coef}')
print(f'Spearman r: {rho_coef}')


# 7. scikit-learn 

`import sklearn `

## 7.1. decomposition

### 7.1.1. FactorAnalysis

```py
from sklearn.decomposition import FactorAnalysis


iris = pd.read_csv('iris_dataset.csv')
X = iris.values
cols = iris.columns.tolist()

factor = FactorAnalysis(n_components=4).fit(X)
factor_comp = np.round(factor.components_,3)

print(pd.DataFrame(factor_comp,columns=cols))

print(f'Explained variance by each component:\n {evr}')

print(pd.DataFrame(pca_comp,columns=cols))

```

### 7.1.2. PCA


```py
homes=pd.read_csv("homes.csv")
X = homes.values
cols = homes.columns.tolist()

import imp
from sklearn.decomposition import PCA

pca = PCA().fit(X)

# 获取每个数据的有效占比  
evr = pca.explained_variance_ratio_

pca_comp = np.round(pca.components_,3)

print("This is the result of the PCA on homes.csv:")
print(pd.DataFrame(pca_comp,columns=cols))

```



## 7.2. manifold

### 7.2.1. TSNE
```py
import imp
from sklearn.manifold import TSNE
tsne = TSNE(init='pca',
            # Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and 50.
            perplexity=50, 
            # For larger values, the space between natural clusters will be larger in the embedded space. Again, the choice of this parameter is not very critical.
            early_exaggeration=25, 
            # Maximum number of iterations for the optimization. Should be at least 250.
            n_iter=300 
            )

Tx = tsne.fit_transform(X)




import numpy as np
import matplotlib.pyplot as plt
plt.xticks([], [])
plt.yticks([], [])
for target in np.unique(ground_truth):
  selection = ground_truth==target
  X1, X2 = Tx[selection, 0], Tx[selection, 1]
  plt.plot(X1, X2, 'o', ms=3)
  c1, c2 = np.median(X1), np.median(X2)
  plt.text(c1, c2, target, fontsize=18, fontweight='bold')

plt.show()

```

## 7.3. preprocessing

### 7.3.1. scale

```py
digits = pd.read_csv('digits.csv')
truth = pd.read_csv('ground_truth.csv')

X = digits.values
ground_truth = truth.values

import imp
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

pca = PCA(n_components=30)
Cx = pca.fit_transform(scale(X))
evr = pca.explained_variance_ratio_

print(f'Explained variance {evr}')

```

## 7.4. cluster

### 7.4.1. DBSCAN



### 7.4.2. KMeans

```py
X = digits.values
ground_truth = truth.values

import imp
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

pca = PCA(n_components=30)
Cx = pca.fit_transform(scale(X))

from sklearn.cluster import KMeans
clustering = KMeans(n_clusters=10, n_init=10, random_state=1)

clustering.fit(Cx)

import numpy as np
ms = np.column_stack((ground_truth,clustering.labels_))
df =  pd.DataFrame(ms, columns = ['Ground truth', 'Clusters'])
ctab = pd.crosstab(df['Ground truth'], df['Clusters'], margins=True)





from sklearn.cluster import KMeans
import numpy as np
inertia = list()
for k in range(1, 21):
  clustering = KMeans(n_clusters=k,
                      n_init=10, random_state=1)
  clustering.fit(Cx)
  inertia.append(clustering.inertia_)
delta_inertia = np.diff(inertia) * (-1)

```


## 7.5. collections

