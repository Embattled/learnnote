- [1. matplotlib包 图表制作](#1-matplotlib包-图表制作)
- [2. matplotlib.pyplot](#2-matplotlibpyplot)
  - [2.1. 基本操作](#21-基本操作)
    - [2.1.1. matplotlib.pyplot.show](#211-matplotlibpyplotshow)
    - [2.1.2. matplotlib.pyplot.subplots 子图](#212-matplotlibpyplotsubplots-子图)
    - [2.1.3. 坐标轴设置 axes](#213-坐标轴设置-axes)
    - [2.1.4. 标签标题设置 label](#214-标签标题设置-label)
  - [2.2. Annotations](#22-annotations)
  - [2.3. 折线图 plot](#23-折线图-plot)
    - [2.3.1. 画图基本](#231-画图基本)
    - [2.3.2. 折线图线的颜色及样式 标识 参数](#232-折线图线的颜色及样式-标识-参数)
  - [2.4. 饼图 pie](#24-饼图-pie)
  - [2.5. bar  条形图](#25-bar--条形图)
  - [2.6. 直方图 Histograms  matplotlib.pyplot.hist](#26-直方图-histograms--matplotlibpyplothist)
    - [2.6.1. 单数据图](#261-单数据图)
    - [2.6.2. 多数据图](#262-多数据图)
  - [2.7. 箱线图  plt.boxplot](#27-箱线图--pltboxplot)
  - [2.8. 散点图 Scatterplots](#28-散点图-scatterplots)
    - [2.8.1. 使用numpy 来分析散点数据](#281-使用numpy-来分析散点数据)
  - [2.9. 图的设置](#29-图的设置)
    - [2.9.1. 为折线图的线添加注解](#291-为折线图的线添加注解)
  - [2.10. Text properties](#210-text-properties)


# 1. matplotlib包 图表制作
Plot show graphically what you've defined numerically

# 2. matplotlib.pyplot

该包是 matplotlib 最核心的包, 也是一般使用者会使用的最多的包

* `import matplotlib.pyplot as plt`  
* 图的数据输入接受 numpy.array, 因此和 pandas 一起使用的时候需要用 `.values`


## 2.1. 基本操作

与图的类型无关, 是所有图通用的相关设置与操作  

### 2.1.1. matplotlib.pyplot.show

`matplotlib.pyplot.show(*, block=None)`  

* 显示所有在程序中 `打开` 的图像  
* block 代表该函数是否阻塞
  * 在非交互模式下, 例如远程Terminal, block 默认为 True, 只有在图像关闭后函数才返回
  * 在交互模式下, block 默认为 True, 立即返回


### 2.1.2. matplotlib.pyplot.subplots 子图

`matplotlib.pyplot.subplot(*args, **kwargs)`  

* 分割图形 , 画多个图  
* 返回值会自动得到用于设置参数的 `axes.Axes` 对象

```py
# 第一个参数 *argsint, (int, int, index), or SubplotSpec, default: (1, 1, 1)
# 221代表分成2行2列后选择第一个子图

# 该函数在调用的时候会自动将下一个画图函数应用到子图中
# 调用完成后直接画就行

ax1 = plt.subplot(221)
plt.plot(...)
ax2 = plt.subplot(222)
plt.plot(...)
ax3 = plt.subplot(223)
plt.plot(...)
ax4 = plt.subplot(224)
plt.plot(...)
# 优雅的写法
ax1 = plt.subplot(2,2,1)




# using the variable ax for single a Axes
fig, ax = plt.subplots()

# using the variable axs for multiple Axes
fig, axs = plt.subplots(2, 2)

# using tuple unpacking for multiple Axes
fig, (ax1, ax2) = plt.subplot(1, 2)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplot(2, 2)
```

### 2.1.3. 坐标轴设置 axes

* axes 是一个图的坐标轴对象的引用
* 如果获取了一个 axes 变量 , 然后就可以通过这个变量来对图标的坐标轴进行设置


图的axes设置:
* grid()      网格设置
* set()       坐标轴范围与长度设置


```py
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

### 2.1.4. 标签标题设置 label

* 不需要调用对象, 直接使用 plt 包的函数
* 坐标轴标签
  * plt.xlabel()
  * plt.ylabel()
* 图标题
  * title()
* 坐标内容
  * plt.xticks()
  * plt.yticks()

```py
# 设置坐标轴标签 , 指明x y轴分别是什么
plt.xlabel('Months')
plt.ylabel('Number of Hurricanes')

# 设置图的标题
plt.title("MLB Players Height")



# matplotlib.pyplot.xticks(ticks=None, labels=None)  
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
## 2.2. Annotations

为图片添加注释
* `matplotlib.pyplot.annotate(text, xy, *ara, **kwargs)`  

参数说明: 
* text : str  , 标注的文字 , 必须参数
* xy   : 是最基础的坐标,必须参数    
* xytext : (float, float), optional  文字的坐标 , 默认就在xy 上 ,若指定了文字坐标, 就可以添加 text 指向 xy 的箭头
* arrowprops : (dict)  用于指定 text 指向 xy 的箭头格式
  * 例: `arrowprops=dict(arrowstyle="->",connectionstyle="arc3`
  * 该参数实际上调用了 `FancyArrowPatch` 的创建, 详细的设置可以参照相关函数
  * 使用 `arrowstype` 可以快速设置内置的箭头格式




arrowpros 箭头格式设置: 
* 如果指定了`'arrowstyle'` , 则可以使用一些默认设定
    * '-' 	None
    * '->' 	head_length=0.4,head_width=0.2
    * '-[' 	widthB=1.0,lengthB=0.2,angleB=None
    * '|-|' 	widthA=1.0,widthB=1.0
    * 'fancy' 	head_length=0.4,head_width=0.4,tail_width=0.4
    * 'simple' 	head_length=0.5,head_width=0.5,tail_width=0.2
    * 'wedge' 	tail_width=0.3,shrink_factor=0.5
* 分为两种 如果字典中没有指明参数`'arrowstyle'` 则是手动模式
    * width 	The width of the arrow in points
    * headwidth 	The width of the base of the arrow head in points
    * headlength 	The length of the arrow head in points
    * shrink 	Fraction of total length to shrink from both ends
    * ? 	Any key to matplotlib.patches.FancyArrowPatch


## 2.3. 折线图 plot

### 2.3.1. 画图基本
```py

# plt.plot( x轴, y轴)
plt.plot(range(5,13),hurricane_table[["2005"]].values)

# 可以运行多次 就可以在同一个图中画多条数据
plt.plot(range(5,13),hurr_table[["2015"]].values)
plt.show()

```
### 2.3.2. 折线图线的颜色及样式 标识 参数 

对于网格和折线都可以对其进行样式设置
```py
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

## 2.4. 饼图 pie


`matplotlib.pyplot.pie(values, colors=colors, labels=labels,explode=explode, autopct='%1.1f%%', shadow=True)` 

* x : array-like 第一个参数, 权重数列 , 会自动根据 x/sum(x) 为每个部位分配大小
* explode : array-like default: None  突出某块部分 , 数组内容是对应每块的偏移位移
* labels : list , 对每块的标签
* colors : array-like, optional, default: None . A sequence of matplotlib color args. 对每块进行指定上色
* autopct : None (default), str, or function, optional .饼块内的标签
* shadow : bool, optional, default: False . Draw a shadow beneath the pie.





## 2.5. bar  条形图
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



## 2.6. 直方图 Histograms  matplotlib.pyplot.hist

### 2.6.1. 单数据图
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
### 2.6.2. 多数据图  



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


## 2.7. 箱线图  plt.boxplot

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

## 2.8. 散点图 Scatterplots

Scatterplots show clusters of data rather than trends (as with line graphs) or discrete values (as with bar charts).  
The purpose of a scatterplot is to help you see data patterns.  

```py

homes_table = pd.io.parsers.read_csv("homes.csv")

x = homes_table[["Sell"]].values
y = homes_table[["Taxes"]].values

plt.scatter(x, y, s=[100], marker='x', c='r')

```

### 2.8.1. 使用numpy 来分析散点数据 


```py
import numpy as np
import matplotlib.pylab as plb

z = np.polyfit(x.flatten(), y.flatten(), 1)
p = np.poly1d(z)

# 注意 y轴 是 p(x)
plb.plot(x, p(x), 'k-')

```


## 2.9. 图的设置


###  2.9.1. 为折线图的线添加注解 

```py
# 两条折线
plt.plot(range(1,13),aritravel[["1958"]].values, ls='-',color='r',marker='s')
plt.plot(range(1,13),aritravel[["1960"]].values, ls='--',color='b',marker='o')

# 创建一个解释 , 指明每条折线分别是什么 , 似乎是按照折线添加的顺序自动链接
# facecolor = None 代表透明
plt.legend(['1958', '1960'], loc=2,facecolor ="None")
# loc 是位置 , 1234分别是从左上角开始的逆时针旋转

```





## 2.10. Text properties

对于一切字符实例,都可以指定其属性  (e.g., title(), xlabel() and text()).
The matplotlib.text.Text instances have a variety of properties which can be configured via keyword arguments to the text commands   

https://matplotlib.org/tutorials/text/text_props.html#sphx-glr-tutorials-text-text-props-py


