- [1. matplotlib包 图表制作](#1-matplotlib包-图表制作)
  - [1.1. matplotlib 的类](#11-matplotlib-的类)
    - [1.1.1. Figure](#111-figure)
    - [1.1.2. Axes](#112-axes)
    - [1.1.3. Axis](#113-axis)
    - [1.1.4. Artist](#114-artist)
  - [1.2. Customizing Matplotlib](#12-customizing-matplotlib)
  - [1.3. 泛用性参数](#13-泛用性参数)
    - [1.3.1. fontdict](#131-fontdict)
- [2. matplotlib.pyplot](#2-matplotlibpyplot)
  - [2.1. plt 基本通用操作](#21-plt-基本通用操作)
    - [2.1.1. show](#211-show)
    - [2.1.2. savefig](#212-savefig)
  - [2.2. 图像创建与全局定义](#22-图像创建与全局定义)
    - [2.2.1. pyplot.figure](#221-pyplotfigure)
    - [2.2.2. pyplot.subplots](#222-pyplotsubplots)
- [3. matplotlib.figure Figure](#3-matplotlibfigure-figure)
  - [3.1. Figure 类](#31-figure-类)
  - [3.2. 添加子图 Axes](#32-添加子图-axes)
    - [3.2.1. fig.add\_axes](#321-figadd_axes)
    - [3.2.2. figure.add\_subplot](#322-figureadd_subplot)
- [4. matplotlib.axes Axes](#4-matplotlibaxes-axes)
  - [4.1. Plotting](#41-plotting)
    - [4.1.1. Basic 基础图](#411-basic-基础图)
      - [4.1.1.1. plot 万物基础-折线图](#4111-plot-万物基础-折线图)
      - [4.1.1.2. bar 条图 直方图](#4112-bar-条图-直方图)
      - [4.1.1.3. lines](#4113-lines)
    - [4.1.2. Spans 跨度线](#412-spans-跨度线)
    - [4.1.3. Spectral](#413-spectral)
    - [4.1.4. Statistics 统计图](#414-statistics-统计图)
    - [4.1.5. Binned 分箱图](#415-binned-分箱图)
      - [4.1.5.1. histogram 标准直方图](#4151-histogram-标准直方图)
      - [4.1.5.2. staris 阶梯图](#4152-staris-阶梯图)
    - [4.1.6. 2D arrays 二维数据](#416-2d-arrays-二维数据)
      - [4.1.6.1. Axes.imshow 图像显示](#4161-axesimshow-图像显示)
    - [4.1.7. Text and annotations 文字和标注](#417-text-and-annotations-文字和标注)
  - [4.2. Axis / limits - 操作数据轴的一些表现](#42-axis--limits---操作数据轴的一些表现)
    - [4.2.1. Axis limits and direction](#421-axis-limits-and-direction)
      - [4.2.1.1. Axes limit](#4211-axes-limit)
      - [4.2.1.2. Axes direction](#4212-axes-direction)
    - [4.2.2. Axes bound](#422-axes-bound)
    - [4.2.3. Axis labels, title, and legend - 图的标题, 坐标轴设置](#423-axis-labels-title-and-legend---图的标题-坐标轴设置)
      - [4.2.3.1. Axis labels - 图坐标轴的标签](#4231-axis-labels---图坐标轴的标签)
      - [4.2.3.2. Axes title - 图的标题](#4232-axes-title---图的标题)
      - [4.2.3.3. legend - 图例说明](#4233-legend---图例说明)
    - [4.2.4. Axis scales - 坐标轴的比例](#424-axis-scales---坐标轴的比例)
    - [4.2.5. Autoscling and margins](#425-autoscling-and-margins)
      - [4.2.5.1. Autoscaling](#4251-autoscaling)
      - [4.2.5.2. margins](#4252-margins)
    - [4.2.6. Aspect ratio](#426-aspect-ratio)
    - [4.2.7. Ticks and tick labels](#427-ticks-and-tick-labels)
  - [4.3. 以前的内容](#43-以前的内容)
    - [4.3.1. 坐标轴设置 axes](#431-坐标轴设置-axes)
    - [4.3.2. 标签标题设置 label](#432-标签标题设置-label)
  - [4.4. Annotations](#44-annotations)
  - [4.5. 折线图 plot](#45-折线图-plot)
    - [4.5.1. 画图基本](#451-画图基本)
    - [4.5.2. 折线图线的颜色及样式 标识 参数](#452-折线图线的颜色及样式-标识-参数)
  - [4.6. 饼图 pie](#46-饼图-pie)
  - [4.7. bar  条形图](#47-bar--条形图)
  - [4.8. 直方图 Histograms  matplotlib.pyplot.hist](#48-直方图-histograms--matplotlibpyplothist)
    - [4.8.1. 单数据图](#481-单数据图)
    - [4.8.2. 多数据图](#482-多数据图)
  - [4.9. 箱线图  plt.boxplot](#49-箱线图--pltboxplot)
  - [4.10. 散点图 Scatterplots](#410-散点图-scatterplots)
    - [4.10.1. 使用numpy 来分析散点数据](#4101-使用numpy-来分析散点数据)
  - [4.11. 图的设置](#411-图的设置)
    - [4.11.1. 为折线图的线添加注解](#4111-为折线图的线添加注解)
  - [4.12. Text properties](#412-text-properties)
- [5. matplotlib.animation](#5-matplotlibanimation)
  - [5.1. Animation](#51-animation)
  - [5.2. Helper Classes](#52-helper-classes)


# 1. matplotlib包 图表制作

Plot show graphically what you've defined numerically  

用类似 Matlab 的方法制作各种图标, 是数据可视化, 数据分析的基础包


## 1.1. matplotlib 的类

plt 包是面向对象思想的库  
整个 plt 的使用流程中, 基本上都是在对具体的某个图片对象进行操作和修改  
![类](https://matplotlib.org/stable/_images/anatomy.png)

整个库最常使用的类从顶层到底层如下: 

![继承关系](https://matplotlib.org/stable/_images/inheritance-7345c0ddb5186802e9fa7f2fb57416ac6be9621b.png)

### 1.1.1. Figure

Figure 是整个图片, Figure 中:
* 保持了对 子Axes的追踪
* 图片的各种 Artists (包括titles, figure legends, colorbars, etc)
* 子图的对象


注意: 大部分的实际使用的函数都是同时创建 figure 和 axes, 但是实际上 axes 是 figure 的一部分, 可以先创建 figure 再往里面添加 axes
* `fig = plt.figure()  # an empty figure with no Axes`
* `fig, ax = plt.subplots()  # a figure with a single Axes`
* `fig, axs = plt.subplots(2, 2)  # a figure with a 2x2 grid of Axes`

### 1.1.2. Axes

Axes 是 Figure 的一个 Artist , 用于控制整个图片的区域, 是所有轴的集合管理对象
* axes 虽然结构上是 Figure 的一部分, 但是却是整个 plt 的核心画图接口 e.g. `ax.plot()`
  * 大部分的画图函数都是作为 axes 的方法被定义的
  * 根据图片的维度个数, 包括了 n 个子 Axis 对象
  * 也包括了对应的 title, x-label, y-label 等  
* Figure 之所以最大, 是因为它还管理了图片的分辨率, 图片的尺寸, 背景的颜色等相对来说 trivial 的属性


### 1.1.3. Axis

名词解释
* tick : the marks on the Axis, 轴的刻度
* ticklabels : strings labeling the ticks, 刻度的标签

Axis 则是 Axes 的一部分 :
* provide ticks and tick labels to provide scales for the data in the Axes
* 用于正确的确定刻度和刻度标签的位置


### 1.1.4. Artist

是整个 matplotlib 库的最基础的类, 祖先类

属于 matplotlib 的专用称呼, 即在 Figure 中所有可见的元素都是 Artist  (even Figure, Axes, and Axis objects)

* 除此之外, includes Text objects, Line2D objects, collections objects, Patch objects, etc.
* 在最后的渲染中, 所有的 Artist 被画在了画布(canvas)上 
* 大部分细则的 Artist 都是绑定在 Axes 上的, 无法做到多个 Axes 之间共享

## 1.2. Customizing Matplotlib

Customizing Matplotlib with style sheets and rcParams


## 1.3. 泛用性参数

### 1.3.1. fontdict

fontdict : `dict` A dictionary controlling the appearance of the title text, the default fontdict is

```py
{'fontsize': rcParams['axes.titlesize'],
 'fontweight': rcParams['axes.titleweight'],
 'color': rcParams['axes.titlecolor'],
 'verticalalignment': 'baseline',
 'horizontalalignment': loc}

```

# 2. matplotlib.pyplot

该包是 matplotlib 最核心的包, 也就是整个画图包提供给普通用户的接口
* `import matplotlib.pyplot as plt`  
* 其他的包的对象都可以通过该类自动生成, 因此在实际使用中不需要导入其他的包, 但是在学习的时候仍然需要参照其他包的文档
* 图的数据输入接受 numpy.array, 因此和 pandas 一起使用的时候需要先转换类型  


该包是一个基于状态的包, 很多对象都是隐式的保存在内存中的, 某种程度上说方便也方便, 但是不利于功能的函数化
matplotlib.pyplot is a state-based interface to matplotlib.


## 2.1. plt 基本通用操作

与图的类型无关, 是所有图通用的相关设置与操作  

* plt.show()      :图像GUI显示
* plt.savefig()   :保存图片


### 2.1.1. show

`matplotlib.pyplot.show(*, block=None)`  

* 显示所有在程序中 `打开` 的图像  
* block 代表该函数是否阻塞
  * 在非交互模式下, 例如远程Terminal, block 默认为 True, 只有在图像关闭后函数才返回
  * 在交互模式下, block 默认为 True, 立即返回

### 2.1.2. savefig

```py
savefig(fname, *, dpi='figure', format=None, metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None, **kwargs
       )
``` 
* `fname`     : str or path-like or binary file-like
* `format`    : str, 决定输出格式, 注意 plt 中如果 format 指定的话, 则完全不会影响 fname, 如果 format 是默认值, 则会从 fname 自动推导图片格式, 默认 Png
* `dpi`       : float or 'figure', 每 inch 是多少分辨率
* `bbox_inches`: str or `Bbox`, 如果是字符串 `tight`, 则会画出图片边界框
* `pad_inches`: 在 bbox_inches 指定为 `tight` 的时候, 指定图片周围的留白宽度



## 2.2. 图像创建与全局定义 

图像创建的本质就是创建 Figure 对象以及为其添加 Axes 对象

使用 matplotlib 创建图形有两种使用 style:
1. 使用面向对象方法, 显式的创建 Figure 对象和 Axes 对象, 然后在其上调用方法, 比较麻烦, 但是清晰
创建并获取图片对象`Figure`和 `axes.Axes`对象
```py
fig = plt.figure()  # an empty figure with no Axes
ax = fig.add_subplot(1, 2, 1)
# some plot function
ax = fig.add_subplot(1, 2, 2)
# some plot function

# 一口气创建所有的 axes 对象  
fig, ax = plt.subplots()  # a figure with a single Axes
fig, axs = plt.subplots(2, 2)  # a figure with a 2x2 grid of Axes

```

或者 直接使用 pyplot 包的懒人函数创建各种图形, 可以非常快, 但是资源不能复用, 这部分的函数需要单独写在一段里  


### 2.2.1. pyplot.figure

```py
matplotlib.pyplot.figure(
    num=None, figsize=None, dpi=None, 
    facecolor=None, edgecolor=None, frameon=True, 
    FigureClass=<class 'matplotlib.figure.Figure'>, 
    clear=False, **kwargs)
```

创建一个新的图 `Figure` 对象, 或者激活一个已经存在的图, 不会直接创建任何 Axes

* `num`       : int or str or `Figure`
  * 作为图在 matplotlib 内部内存的标识器, 分为数字版本 `Figure.number` 和字符串版本
  * 如果是 None, 则会自动从 1 编号
  * 如果是数字或者字符串, 则相应的创建并赋予标号
  * 如果该标识的图片已经存在, 则 this figure is made active and returned. 通过 clear 参数可以覆盖的重新创建
* `figsize`   :(float, float), (default: [6.4, 4.8]) Width, height in inches.
* `dpi`       : float, default 100.0 . The resolution of the figure in dots-per-inch.
* `facecolor` : color, 默认为白色. The background color.
* `edgecolor` : color, 边框颜色. The border color.
* `frameon`   : bool, default: True. If False, suppress drawing the figure frame.
* `FigureClass`: subclass of Figure. If set, an instance of this subclass will be created, rather than a plain Figure.
* `clear`     : bool, default: False. If True and the figure already exists, then it is cleared.
* `layout`    : `{'constrained', 'tight', LayoutEngine, None}`, default: None. 需要参照 Figure 类的文档
* `**kwargs`  : 其他关键字参数传递给 Figure 类的构造函数
* Returns: `Figure`


### 2.2.2. pyplot.subplots

```py
matplotlib.pyplot.subplots(
  nrows=1, ncols=1, *, 
  sharex=False, sharey=False, 
  squeeze=True, 
  width_ratios=None, height_ratios=None, 
  subplot_kw=None, gridspec_kw=None, 
  **fig_kw)
```
Create a figure and a set of subplots.
* `nrows`, `ncols`  :int, default: 1 .Number of rows/columns of the subplot grid.
* `sharex`, `sharey` : bool or `{'none', 'all', 'row', 'col'}`, default: False.
  * Controls sharing of properties among x (sharex) or y (sharey) axes:
  * `True` or `all` 会在所有子图中共享 x/y 轴
  * `False` or `none` 会让各个子图的坐标轴互相独立
  * `row` : 子图的每行里会共享 x/y 轴
  * `col` : 子图的每列会共享 x/y 轴
  * 如果开启了坐标轴共享的话:
    * When subplots have a shared x-axis along a column, only the x tick labels of the bottom subplot are created.
    * When subplots have a shared y-axis along a row, only the y tick labels of the first column subplot are created.
    * To later turn other subplots' ticklabels on, use tick_params.
    * When subplots have a shared axis that has units, calling set_units will update each axis with the new units.
* `squeeze` :bool, default: True. 正常来说返回的 Axes 应该是个2维矩阵, 但是 squeeze 为 True 的话会压缩掉shape为 1 的维度
  * if only one subplot is constructed (nrows=ncols=1), the resulting single Axes object is returned as a scalar
  * for Nx1 or 1xM subplots, the returned object is a 1D numpy object array of Axes objects.
* `width_ratios` `height_ratios` : array-like of length ncols, optional. 
  * 是一个从 `gridspec_kw` 提取的便捷参数
  * 用于指定图之间的相互比例, 通过比例来调整子图的大小
  * Each column gets a relative width of `width_ratios[i] / sum(width_ratios) `
  * Each row gets a relative height of `height_ratios[i] / sum(height_ratios)`
* `subplot_kw` : 用于传递给 `figure.add_subplot()` 的关键字参数字典
* `gridspec_kw`: 用于传递给 `GridSpec` constructor 的关键字参数字典
* `**fig_kw`   : 其余的关键字参数都是传递给 `pyplot.figure()` 的
* return:
  * fig : `Figure`
  * ax  : `Axes` or array of Axes  


# 3. matplotlib.figure Figure

Figure 在 matplotlib 中是最高级的类, `Top level Artist`, 保存了对所有图片中元素的链接  

在 matplotlib.figure 模组中, 实现的内容包括
* Figure 类, 即最高级别的 Artist  `class matplotlib.figure.Figure()`
* SubFigure , 嵌入式的 子图类
* SubplotParams , 用于控制子图类默认的空间

对于用户来说, 创建 figure 一般通过 pyplot 的接口
* figure
* subplots
* subplot_mosaic

对于开发一个嵌入在应用程序内的图表来说, 可能需要手动实例化一个 figure 类

## 3.1. Figure 类

```py
class matplotlib.figure.Figure(
  figsize=None, dpi=None, *, 
  facecolor=None, edgecolor=None, 
  linewidth=0.0, 
  frameon=None, 
  subplotpars=None, 
  tight_layout=None, 
  constrained_layout=None, layout=None, **kwargs)
```


## 3.2. 添加子图 Axes

### 3.2.1. fig.add_axes

Add an Axes to the figure.
函数原型 `add_axes(*args, **kwargs)`  
```py
add_axes(rect, projection=None, polar=False, **kwargs)
add_axes(ax)
```
参数意思:  
* rect : tuple (left, bottom, width, height).  
  * 用于指定新 Axes 在figure中的位置, 所有参数都用 0~1 小数来表示在 figure 中的相对位置/宽度
  *  


### 3.2.2. figure.add_subplot

Add an Axes to the figure as part of a subplot arrangement.

函数原型为: `add_subplot(*args, **kwargs)`, 意为该函数有多种调用方法
```py
add_subplot(nrows, ncols, index, **kwargs)
add_subplot(pos, **kwargs)
add_subplot(ax)
add_subplot()
```

参数意思:
* `*args` : 多种类参数, `int, (int, int, index), or SubplotSpec, default: (1, 1, 1)`
  * The position of the subplot described by one of
  * (int, int, index), 前两个参数指定具体的网格 nrows, ncols, 第三个 index 用于指定该新添加的 Axes 的位置, index 可以是单个数字意为从左到右的Index, 也可以是一个 2-tuple.  
  * pos : 一个 3-digit integer, 用于最简便的替换(int, int, index), 因此该方法无法对应子图个数大于9的情况
  * `SubplotSpec` : 一个子图实例, 即在别处已经创建好的 Axes 直接传入图中.
* return :  `axes.SubplotBase`, or another subclass of `Axes`
* `kwargs` : 传递给 class `Axes` 构造函数的参数


# 4. matplotlib.axes Axes

参考官方文档 `matplotlib.axes` 

是具体的画数据的类, 是控制画图的核心

继承上来看 `artist.Artist -> axes._base._AxesBase -> axes._axes.Axes`

通常来说, 一个二维图的 axes 里有 2个 axis  

整个模组里主要就是关于 `Axes` 类的操作


## 4.1. Plotting

各种以 Axes 对象方法来操作画图的函数, 包括了matplotlib.pyplot 的各种画图函数的真实画图接口方法

* 这些函数的返回值都是各种 `具体图的对象`, 例如 plot 会返回 `matplotlib.lines.Line2D` 的对象
* 一般来说不需要直接操作图的属性, 转而用方法的 `**kwargs` 参数即可
* 不同图的类也不同, 意味着不同种类的图可操作的属性也不同 , 因为太多了所以各种图的函数的 `**kwargs` 就暂且省去不看了





### 4.1.1. Basic 基础图

最基础的图, 也是最常用的包括

| 类方法   | 图类型                                                           |
| -------- | ---------------------------------------------------------------- |
| plot     | 折线图, Plot y versus x as lines and/or markers.                 |
| errorbar | Plot y versus x as lines and/or markers with attached errorbars. |
| bar      | 条形图                                                           |
| pie      | 饼图                                                             |

#### 4.1.1.1. plot 万物基础-折线图 

`Axes.plot(*args, scalex=True, scaley=True, data=None, **kwargs)`
* 最主要的数据是通过无限参数 args 来输入的, 具体的使用方法如下
* `plot([x], y, [fmt], *, data=None, **kwargs)`
* `plot([x], y, [fmt], [x2], y2, [fmt2], ..., **kwargs)`
* `fmt` : 是一个便捷参数, 用于省去通过类方法或者属性来修改图的样子的过程, 直接通过一个字符串来定义图的样式

使用例
```py
plot(x, y)        # plot x and y using default line style and color
plot(x, y, 'bo')  # plot x and y using blue circle markers
plot(y)           # plot y using x as index array 0..N-1
plot(y, 'r+')     # ditto, but with red plusses
```


`data = None` 传入类字典的数据来便捷的指定数据
* 比起直接传入 x , y 的数据, 转而传入 data 的dict对象的对应 keys
* `plot('xlabel', 'ylabel', data=obj)` 

画出多个图
* 由于 matplotlib 是基于状态的库 所以直接依次多次调用 plot 即可在同一张图片上画多个折现
* 可以直接将 x, 或者 y 变成多维, 来直接传入多个曲线, 这里注意, matplotlib 的坐标轴维度(属性)在前, 数据index在后
  * 若单条数据有 N 个点, 共计 m 条数据 即画出 m 条折线, 则 y 的维度应该是 (N,m) 而不是 (m,N), 这与一些数据分析的维度顺序相反
  *  If both x and y are 2D, they must have the same shape.
  *  If only one of them is 2D with shape (N, m) the other must have length N and will be used for every data set m.
* 也可以使用 `*args` 的特性, 依次传入多个图的 `x,y,fmt` 即可

```py
# 多次调用
plot(x1, y1, 'bo')
plot(x2, y2, 'go')

# 高维 x or y
x = [1, 2, 3]
y = np.array([[1, 2], [3, 4], [5, 6]])
plot(x, y)

# 无限参数
plot(x1, y1, 'g^', x2, y2, 'g-')
```

#### 4.1.1.2. bar 条图 直方图

`Axes.bar(x, height, width=0.8, bottom=None, *, align='center', data=None, **kwargs)`
* 条状图, x 制定了条的对齐点, 由高度和 width 来指定条的大小
* 许多参数都可以以 list 的形式传入, 用于给每一个条单独指定大小
* x : float or array-like
* height : float or array-like
  * 如果 bottom 是有单位的值, 那么要注意 height 值应该是与 bottom 相同单位的 差值
  * (不太理解, 好像是在 time 数据时候需要注意)
* width : float or array-like, `default: 0.8`
  * bar 的宽度, 比较重要
  * 同样的, 如果 x 是有单位的值, width 也应该是与 x 相同单位的, 与 x 值的差值
* bottom: float or array-like, `default: 0`
  * 底部的坐标
  * 用于时间轴的表示可能很有用
* align : {'center', 'edge'}, default: 'center'
  * x 的对齐位置
  * 默认对其的是左边, 如果要对其右边的话 width 传入对应的负值即可




#### 4.1.1.3. lines

画横线/竖线
* 这里线的长度是根据坐标轴的数值来指定的, 且为必须参数
* 如果想方便快捷的画跨越整个图的直线, 直接使用 `ax*line` 
* 该函数也可以转换成类似于 `ax*line` 的使用方法  
  * `By using ``transform=vax.get_xaxis_transform()`` the y coordinates are scaled`
  * ` such that 0 maps to the bottom of the axes and 1 to the top.`
* 该函数一次可以画多条线 : 返回 `LineCollection` 对象

```py
Axes.vlines(x, ymin, ymax, colors=None, linestyles='solid', label='', *, data=None, **kwargs)
Axes.hlines(y, xmin, xmax, colors=None, linestyles='solid', label='', *, data=None, **kwargs)

Returns:
    LineCollection

vax.vlines([1, 2], 0, 1, transform=vax.get_xaxis_transform(), colors='r')

```

### 4.1.2. Spans 跨度线

主要用来做一些标记, 该分类提供了一些在图中画水平线/垂直线的函数, 功能上似乎和 basic 中的 vlines 没什么区别, 但是这里线的长度是相对于整个图, 与数据内容无关, 且该函数一次只能画一条线/一个多边形  

* line 函数本质上与 plot 相同都是向图中添加 `Line2D` 对象 和一个 transform
* span 函数则是向图中添加 `Polygon`
* 因此查阅各个函数的 kwargs 内容即参照以上的对象即可

画单线的函数
* Axes.axhline   Add a horizontal line across the Axes.
* Axes.axvline   Add a vertical line across the Axes.
* Axes.axline    Add an infinitely long straight line.
  * 指定两点画一条线
  * 或者指定一点 + `slope` 来画一条线
  * xy 代表一个点, 为 `(float,float)` 的格式

```py
Axes.axhline(y=0, xmin=0, xmax=1, **kwargs)
Axes.axvline(x=0, ymin=0, ymax=1, **kwargs)
Axes.axline(xy1, xy2=None, *, slope=None, **kwargs)

Returns:
    Line2D
```


画范围的函数
* Axes.axhspan   Add a horizontal span (rectangle) across the Axes.
* Axes.axvspan   Add a vertical span (rectangle) across the Axes.

```py
Axes.axhspan(ymin, ymax, xmin=0, xmax=1, **kwargs)
Axes.axvspan(xmin, xmax, ymin=0, ymax=1, **kwargs)

Returns:
    Polygon
        # Vertical span (rectangle) from (xmin, ymin) to (xmax, ymax).


```


### 4.1.3. Spectral

### 4.1.4. Statistics 统计图

### 4.1.5. Binned 分箱图

最经典的直方图定义在这里  


#### 4.1.5.1. histogram 标准直方图

```py
Axes.hist(x, bins=None, range=None, density=False, weights=None, 
    cumulative=False, bottom=None, histtype='bar', align='mid', 
    orientation='vertical', 
    rwidth=None, log=False, color=None, 
    label=None, stacked=False, *, data=None, **kwargs)

```

以 numpy 的 histogram 函数作为基础, 配合 matplotplot 库的 BarContainer 或 Ploygon 模块来实现的直方图, 绝大多数参数的意思都继承了 numpy 的 histogram 函数
* 注意, 该函数更倾向是一个统合的便捷函数, 如果数据已经经过 numpy.histogram 进行统计, 则可以使用 `bar` 或者 `stairs` 来画图
* 或者使用使用特殊的参数来将直方图统计的结果直接输入 `Axes.hist` 即 `plt.hist(bins[:-1], bins, weights=counts)`


#### 4.1.5.2. staris 阶梯图

作为直方图的一部分, 将每个 y值数据 画成从 x 轴出发的长方形  
A stepwise constant function as a line with bounding edges or a filled plot.

`Axes.stairs(values, edges=None, *, orientation='vertical', baseline=0, fill=False, data=None, **kwargs)`
* `values` : array-like, The step heights.
* `edges`  : array-like, The edge positions, with `len(edges) == len(vals) + 1`, between which the curve takes on vals values.
* `orientation` : `{'vertical', 'horizontal'}`, default: 'vertical'. 决定了数据的长方体画的方向。
* `baseline` : float, array-like or None, default: 0. 直方图的底部基线数值.
* `fill`   : bool, default: False. 每太看懂, Whether the area under the step curve should be filled.
* `data`   : indexable object, optional. 经典字典型数据输入, If given, all parameters also accept a string `s`, which is interpreted as `data[s]`
* `**kwargs` :StepPatch properties

返回值:
* Returns: `matplotlib.patches.StepPatch` 

### 4.1.6. 2D arrays 二维数据

直接将2为数据画成图的函数

#### 4.1.6.1. Axes.imshow 图像显示

Display data as an image, i.e., on a 2D regular raster.  

该函数的参数原型非常复杂  
`Axes.imshow(X, cmap=None, norm=None, *, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None, origin=None, extent=None, interpolation_stage=None, filternorm=True, filterrad=4.0, resample=None, url=None, data=None, **kwargs)`

* X : array-like or PIL image. 支持直接传入 PIL.image 对象.
  * 支持 (M,N) 2-D 数据, 并可以通过 colormap(`cmap`) 和 normalization(`norm`) 参数来彩色化
  * 支持 (M,N,3) 和 (M,N,4) 图像的直接显示, 数据支持 8bit 和 0~1 小数 

### 4.1.7. Text and annotations 文字和标注


## 4.2. Axis / limits - 操作数据轴的一些表现 

坐标轴, limits  

用来操作数据轴的一些函数, 如同 matplotlib 的类的包容图, 一个 axes 对象管理了两个 axis 对象

虽然一般情况下不需要直接进行 axis 对象的操作, 但是获取对象的方式是存在的
```py
# The use of this function is discouraged. You should instead directly access the attribute

Axes.get_xaxis  # [Discouraged] Return the XAxis instance.
Axes.get_yaxis  # [Discouraged] Return the YAxis instance.

ax.xaxis
ax.yaxis
```

### 4.2.1. Axis limits and direction

操纵坐标轴的 limits

#### 4.2.1.1. Axes limit

xlim ylim 的值, 即在图上显示的坐标区间, 会对数据进行截断, 即传入了 plot 但是不会显示在图上

* `Axes.set_xlim` : Set the x-axis view limits.
  * `(left=None, right=None, *, emit=True, auto=False, xmin=None, xmax=None)`
  * left, right. float,  lim in data coordinates.
  * left and right xlims may also be passed as the tuple `(left, right)` as the first positional argument.
* `Axes.set_ylim` : Set the y-axis view limits.
  * `(bottom=None, top=None, *, emit=True, auto=False, ymin=None, ymax=None)`
* `Axes.get_xlim` : Return the x-axis view limits.
* `Axes.get_ylim` : Return the y-axis view limits.



#### 4.2.1.2. Axes direction
调整坐标轴的方向, 通过这个接口可以反转坐标轴的增加方向, 正常方向是 x 轴向右增加, y 轴向上增加

* `Axes.invert_xaxis()`   调转 x-axis
* `Axes.invert_yaxis()`
* `Axes.xaxis_inverted()`  返回目前的调转状态
* `Axes.yaxis_inverted()`

### 4.2.2. Axes bound




### 4.2.3. Axis labels, title, and legend - 图的标题, 坐标轴设置  

对于一个图来说不可或缺的说明内容, 尤其是当单个图例有多条数据线的时候  

#### 4.2.3.1. Axis labels - 图坐标轴的标签

用于后手指定 x,y 的坐标轴

* `Axes.set_xlabel(xlabel, fontdict=None, labelpad=None, *, loc=None, **kwargs)`   Set the label for the x-axis.
* `Axes.set_ylabel(ylabel, fontdict=None, labelpad=None, *, loc=None, **kwargs)`   Set the label for the y-axis.
  * labelpad 用于指定某种间距, 不太清楚, 传入 `float` 
  * loc  用于指定坐标轴的 label 位于坐标轴的位置, 可以传入  `{'left', 'center', 'right'}`

* `Axes.get_xlabel()`   Get the xlabel text string.
* `Axes.get_ylabel()`   Get the ylabel text string.

#### 4.2.3.2. Axes title - 图的标题

一个图不能没有标题, 这里指定的是 axes 的 title, 理论上如果一个 figure 里只有一个子图的话和 figure 里的指定并没有区别, 应该主要用于有多个子图的情况

这里一个 axes 事实上可以指定最多3个 title

* `Axes.set_title(label, fontdict=None, loc=None, pad=None, *, y=None, **kwargs)`          Set a title for the Axes.
  * loc : `{'center', 'left', 'right'}` 默认 center
  * y : Vertical Axes location for the title (1.0 is the top), 如果是默认值 None 的话则代表自动推测 (自动推测防止覆盖到坐标轴的位置)
  * pad : The offset of the title from the top of the Axes, in points. float, default `6.0`
  * return : matplotlib 自己的文本对象  `matplotlib.text`

* `Axes.get_title(loc='center')`          Get an Axes title. 参数用于指定要获取哪一个 title
  * loc : `{'center', 'left', 'right'}`, str, default: 'center'
 


#### 4.2.3.3. legend - 图例说明

起码在拥有多条折线的图例, 各个线的图例是不可或缺的   

`Axes.legend(*args, **kwargs)    Place a legend on the Axes.`
* `handles` sequence of Artist, optional
* `label` list of str, optional
* return : legend 对象, `matplotlib.legend`
* 还有其他对于 legend box 的各种样式/位置指定参数, 可以直接去参照 `matplotlib.legend` 的文档, 或者日后要用的时候在参照该函数的文档

创建一个易读的 legend 的方法如下
* 1. Automatic detection of elements to be shown in the legend
  * 这需要为每一个折线指定好 label
  * 可以在创建一个折线的时候指定 `ax.plot([1, 2, 3], label='Inline label')`
  * 可以在创建折现后, 通过折线的对象来设置 `line = ax.plot([1, 2, 3])   line.set_label('Label via method')`
* 2. 直接将折线对象和对应的 legend 传入即可
  * 这需要在创建折线的时候保留折线对象
  * 向函数依次传入对应的 折线list 和 legend list
  * 该函数可以保留一部分折线没有 legend, 即只选定想要有 legend 的折线
* 3. 12的结合方法
  * 创建折线的时候指定 label, 并保存对象
  * 向函数传入关键字参数 `handles=[line1,line2]`
* 4. 不安全的懒人方法
  * 直接向 legend 函数传入按照次序的 legend list
  * 会按照折线的创建顺序依次把 legend list 的名称赋予折线
  * matplotlib 官方不推荐这种用法


具体反应在代码上如下
```py
# 方法1 , 设置好 label 后直接调用 legend() 即可
legend()

# 方法2 , 
legend(handles, labels)

# 方法3
legend(handles=handles)

# 方法4
legend(labels)
```

* `Axes.get_legend()`   返回当前 axes 的 legend 对象
* `Axes.get_legend_handles_labels(legend_handler_map=None)` 返回当前 axes 的 legend 对象的 handles 和 labels

```py
# ax.legend() is equivalent to
h, l = ax.get_legend_handles_labels()
ax.legend(h, l)
```

### 4.2.4. Axis scales - 坐标轴的比例  

设置坐标轴 axis 的 `scale`, 例如坐标轴的坐标是指数增长之类的  

设置函数: `(value, **kwargs)` 
* Axes.set_xscale  : Set the xaxis' scale.
* Axes.set_yscale  : Set the yaxis' scale.
* value `{"linear", "log", "symlog", "logit", ...}` or ScaleBase 

获取函数 : 无参数, 返回 axis scale as str
* Axes.get_xscale  : Return the xaxis' scale (as a str).
* Axes.get_yscale  : Return the yaxis' scale (as a str).

### 4.2.5. Autoscling and margins

xlim ylim 的值, 即在图上显示的坐标区间, 自动调整的功能能被称作 Autoscaling  

margin 则是 :  所输入的 Data 的范围再左右扩充, 即一定程度上的留白, 默认的 lim 是 5%.

`Axes.use_sticky_edges` Axes类的一个属性 : When autoscaling, whether to obey all Artist.sticky_edges.
* 默认值为 True
* Setting this to False ensures that the specified margins will be applied, even if the plot includes an image

#### 4.2.5.1. Autoscaling

* `Axes.autoscale(enable=True, axis='both', tight=None)`
  * Autoscale the axis view to the data (toggle).
  * enable: True, False, None. 分别起到 打开, 关闭, 保持 autoscale 机能的作用
  * axis  : {'both','x','y'}, 指定要操作的 autoscale 的坐标轴
  * tight : bool or None, 如果参数为 True, 则会设置 margins 为 0.
    * 不管该参数的具体的值, 都会调用一次 autoscale_view 并将该参数值传递过去
  * `autoscale()` 快速调用来启动 autoscale, 一般用于在设置了 xlim ylim 以后重新启动

* `Axes.autoscale_view(tight=None, scalex=True, scaley=True)`
  * Autoscale the view limits using the data limits.
  * 可以理解为具体执行画图 autoscale 的接口函数
  * 如果 xlim ylim 被设置, 那么该函数调用了也不会起作用
  * `tight` : 
    * True, 根据当前的 margin 的值, 来 autoscale 一次坐标轴的 lim
    * False, 会参照 `axes.autolimit_mode` 的值, 如果后者是 `round_numbers`, 那么会通过 axis major locator 进一步扩充 axis limit, 否则仍然等同于 True
    * None, 维持上次调用的值, 初次调用相当于 False, 结合 `axes.autolimit_mode` 的默认值, 最终的表现相当于执行 True
  * 如果数据被更新了, 则调用 `Axes.relim()` 来重新计算 limit
* `Axes.relim(visible_only=False)`
  * Recompute the data limits based on current artists.
  * `visible_only` : bool, default: False, Whether to exclude invisible artists.




状态获取函数 get :
* `Axes.get_autoscale_on()` : 如果两个轴都打开了 autoscale ,返回 True, 否则返回 False
* `Axes.get_autoscalex_on()`: 返回 xaxis 的状态
* `Axes.get_autoscaley_on()`: 返回 yaxis 的状态

设置函数 set :
* `Axes.set_autoscale_on(b)` : 同时设置两个轴
* `Axes.set_autoscaley_on(b)` 
* `Axes.set_autoscalex_on(b)` 



#### 4.2.5.2. margins 


注意: 一些特殊的 plot, 例如 created with Axes.imshow 的假色彩图像, 会不受 margins 的影响
* 由于 Sticky edges , 特性的存在, 图片 plot 的边缘会粘着坐标轴
* 因此 正值的 margin 不会起作用, 即不管怎么样都不会有 margin , 但是 负值的 margin 仍然生效
  * 对于这种特殊情况可以使用 `Axes.use_sticky_edges = False` 来关闭该特性, 使图片类Plot可以正常添加 margin
 

* `Axes.margins(*margins, x=None, y=None, tight=True)`
  * 无参数调用, 获取坐标轴的 margins, 默认的 margins 是 0.05 即左右各有5% 的留白
  * 通过位置参数或者关键字参数来设置, 只能选择一种设置方法.
  * All input parameters must be floats within the range [0, 1].
  * `*margins` : float, optional.
    *  single positional argument is provided, it specifies both margins of the x-axis and y-axis limits.
    *  two positional arguments are provided, they will be interpreted as xmargin, ymargin.
    *  该方法不能单独修改一个坐标轴的 margin 同时维持另一个坐标轴的 margin 不变, 如果有这种需求, 使用 kw 参数
  * `x, y` : float, optional. 关键字参数.
  * `tight` : bool or None, default: True.
    * 传递给 `Axes.autoscale_view()` 的参数
* `Axes.set_xmargin(m)`
* `Axes.set_ymargin(m)`
  * 单独设置某个坐标轴的 margin, m 参数的意思与上面相同, 都是 `[-0.5,+00]`范围的小数



### 4.2.6. Aspect ratio

设置 Axes 的比例  

* adjustable : how the Axes adjusts to achieve the required aspect ratio.
  * `Axes.set_adjustable(adjustable, share=False)`
    * adjustable : `{'box', 'datalim'}`
    * 如果是 `box`, 则改变具体的 figure 上的表示, 对数据进行拉伸, 如果是 `datalim`, 则会修改数据的 limit
    * share: bool, default False. If True, apply the settings to all shared Axes.
  * `Axes.get_adjustable()` 
    * 返回当前的 adjustable 设置

* aspect :  ratio of the axes scaling. 
  * `Axes.set_aspect(aspect, adjustable=None, anchor=None, share=False)`
    * aspect : `{'auto', 'equal'} or float`
      * `auto` 会填充整个 rectangle
      * `equal` : 相当于 1, 即 x, y 有相同的比例
      * `float` : height/width
    * `adjustable`: `None or {'box', 'datalim'}`, optional
    * `anchor` : `None or str or (float, float)`, optional
      * 如果设置了非 auto 的 aspect, 那么对于 axes 的留白空间, 如何进行定位
      * 可以设置居中或者 四个角落 或者贴着四个边缘居中
      * 具体的输入参照 `axes.set_anchor` 函数
    * `share`: bool, default: False . If True, apply the settings to all shared Axes.

  * `Axes.get_aspect()` : Return the aspect ratio of the axes scaling.
    * This is either "auto" or a float giving the ratio of y/x-scale.


* box_aspect : ratio of height to width. 主要是在 figure 中的比例, 与 data 是无关的.
  * 设置该属性设置函数会将`adjustable` 属性为 `datalim`
  * `Axes.set_box_aspect(aspect=None)` :设置 axes 的比例
    * aspect : `float or None`,  height/width. 
    * 传入 None, 会关闭 fixed box aspect, 使得 height 和 weight 会各自独立进行选择 
  * `Axes.get_box_aspect()` : 返回当前的比例, 默认都是 None

* `Axes.apply_aspect(position=None)` : 根据当前的 aspect 设置来重新计算(adjust) axes
  * Depending on `get_adjustable` this will modify
    *  the Axes box (position), 根据 `get_anchor` 来确定最终的效果
    *  view limits.


### 4.2.7. Ticks and tick labels

坐标刻度管理  

## 4.3. 以前的内容

### 4.3.1. 坐标轴设置 axes

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


### 4.3.2. 标签标题设置 label

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


## 4.4. Annotations

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


## 4.5. 折线图 plot

### 4.5.1. 画图基本
```py

# plt.plot( x轴, y轴)
plt.plot(range(5,13),hurricane_table[["2005"]].values)

# 可以运行多次 就可以在同一个图中画多条数据
plt.plot(range(5,13),hurr_table[["2015"]].values)
plt.show()

```
### 4.5.2. 折线图线的颜色及样式 标识 参数 

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

## 4.6. 饼图 pie


`matplotlib.pyplot.pie(values, colors=colors, labels=labels,explode=explode, autopct='%1.1f%%', shadow=True)` 

* x : array-like 第一个参数, 权重数列 , 会自动根据 x/sum(x) 为每个部位分配大小
* explode : array-like default: None  突出某块部分 , 数组内容是对应每块的偏移位移
* labels : list , 对每块的标签
* colors : array-like, optional, default: None . A sequence of matplotlib color args. 对每块进行指定上色
* autopct : None (default), str, or function, optional .饼块内的标签
* shadow : bool, optional, default: False . Draw a shadow beneath the pie.





## 4.7. bar  条形图
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



## 4.8. 直方图 Histograms  matplotlib.pyplot.hist

### 4.8.1. 单数据图
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
### 4.8.2. 多数据图  



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


## 4.9. 箱线图  plt.boxplot

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

## 4.10. 散点图 Scatterplots

Scatterplots show clusters of data rather than trends (as with line graphs) or discrete values (as with bar charts).  
The purpose of a scatterplot is to help you see data patterns.  

```py

homes_table = pd.io.parsers.read_csv("homes.csv")

x = homes_table[["Sell"]].values
y = homes_table[["Taxes"]].values

plt.scatter(x, y, s=[100], marker='x', c='r')

```

### 4.10.1. 使用numpy 来分析散点数据 


```py
import numpy as np
import matplotlib.pylab as plb

z = np.polyfit(x.flatten(), y.flatten(), 1)
p = np.poly1d(z)

# 注意 y轴 是 p(x)
plb.plot(x, p(x), 'k-')

```


## 4.11. 图的设置


###  4.11.1. 为折线图的线添加注解 

```py
# 两条折线
plt.plot(range(1,13),aritravel[["1958"]].values, ls='-',color='r',marker='s')
plt.plot(range(1,13),aritravel[["1960"]].values, ls='--',color='b',marker='o')

# 创建一个解释 , 指明每条折线分别是什么 , 似乎是按照折线添加的顺序自动链接
# facecolor = None 代表透明
plt.legend(['1958', '1960'], loc=2,facecolor ="None")
# loc 是位置 , 1234分别是从左上角开始的逆时针旋转

```





## 4.12. Text properties

对于一切字符实例,都可以指定其属性  (e.g., title(), xlabel() and text()).
The matplotlib.text.Text instances have a variety of properties which can be configured via keyword arguments to the text commands   

https://matplotlib.org/tutorials/text/text_props.html#sphx-glr-tutorials-text-text-props-py



# 5. matplotlib.animation

使用 animation 模组来创建图的动画  

## 5.1. Animation

提供了 Animation 类用于以最简单的方法来创建动画, 定义了3个类  
* `Animation` : 动画的基类, 并不是主要的操作 interface
* `FuncAnimation` : 基于重复调用一个 func 而形成的动画
* `ArtistAnimation` : 基于 fixed set of `Artist` objects 所构成的类  







## 5.2. Helper Classes

非用户操作接口, 而是各种功能的底层实现



