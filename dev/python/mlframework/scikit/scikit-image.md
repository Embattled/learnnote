# 1. scikit-image

A collection of algorithms for image processing.  
scikit-的图像处理相关库.  

Submodules:
* skimage
    * color
    * data
    * draw
    * exposure
    * feature
    * filters
    * future
    * graph
    * io
    * measure
    * metrics
    * morphology
    * registration
    * restoration
    * segmentation
    * transform
    * util
    * viewer

# 2. io

Utilities to read and write images in various formats.  
sci 的图像 IO 包  

## 2.1. class 类

IO包中提供了操纵数据的几个类.  

### 2.1.1. ImageCollection

有点类似于 DataLoader  

```py
class skimage.io.ImageCollection(
  load_pattern, # Pattern string 或者 list of 文件path , 这个path 可以是相对路径也可以是绝对
  conserve_memory=True,  # 节省内存
  load_func=None, # Default 是 imread, 即标准读取
  **load_func_kwargs)

# 使用 pattern string 来创建
coll = io.ImageCollection(data_dir + '/chess*.png')
len(coll) # = 2 
```
类的属性
* `files` : 文件路径的 list
类的方法
* `concatenate` : 将所有图片作为一个 np.ndarray 返回  


### 2.1.2. MultiImage

* 基于 ImageCollection 的子类  
* 特点是可以读取动图 eg. tif 文件
* 会将所有帧区分存储  


```py
class skimage.io.MultiImage(
  load_pattern, 
  conserve_memory=True, 
  dtype=None, 
  **imread_kwargs)
```


## 2.2. imread 

读一个图像, 读取后的类型是 `ndarray`  

```py
skimage.io.imread(
  fname,   # 图像路径, 也可以是 url
  as_gray=False,  # 是否将图像转为灰度图 64-bit floats
  plugin=None, 
  **plugin_args)
```

## 2.3. imread_collection

读入一组图像, 需要和该库中的 imageCollection 类型配合  

```py
skimage.io.imread_collection(
  load_pattern, # List of objects to load. 一般都是文件名
  conserve_memory=True, # 如果 True, 只会在内存中存储当前使用的图像, 否则会把所有图片读入内存
  plugin=None, **plugin_args)
```


# 3. transform

## 3.1. resize

```py
skimage.transform.resize(
  image, 
  output_shape, 
  order=None, 
  mode='reflect', 
  cval=0, 
  clip=True, 
  preserve_range=False, 
  anti_aliasing=None, 
  anti_aliasing_sigma=None)
""" 
image         : (ndarray) Input image.
output_shape  : tuple or ndarray 
                Size of the generated output image (rows, cols[, …][, dim])
其他7个参数都是通用参数
"""
```

# 4. morphology

形态学变换的专用库

返回值:  
* uint8 array, same shape and type as image. 形态学变换的结果

通用参数:  
* image: ndarry. 输入图像
* footprint: ndarry,opt. 用于形态学变换的匹配图. structuring element.
  * The neighborhood expressed as an array of 1’s and 0’s.
  * If None, use `cross-shaped footprint` (connectivity=1).
  * 注意该参数在不同版本的名称是不同的, 旧版本要求是 2-D ndarray
* out: ndarry, opt. 用于非返回值方式的结果获取.
  * If None is passed, a new array will be allocated.
* shift_x, shift_y: bool, opt. 设定是否平移中心点. 用于对应特殊的 footprint.

## 4.1. generate footprint 

用于快速生成一些 template  

2D template :  
* `skimage.morphology.disk(radius, dtype=<class 'numpy.uint8'>)`
  * radius: int
  * 专门用于生成一种圆盘形状的 footprint, 可以指定半径
  * 各个点到原点的欧几里得距离不会大于 (<=) radius
  * 返回的ndarray 是 (2*radius+1)^2 长度的 0,1 列
* `skimage.morphology.square(width, dtype=<class 'numpy.uint8'>)`
  * width: int
  * 用于生成一个标准正方形的 footprint, 指定宽度
  * radious = floor(width/2), 各个点到原点的棋盘距离不会大于 radious
  * 其实就是直接返回一个 `np.ones((width, width))`
* `skimage.morphology.diamond(radius, dtype=<class 'numpy.uint8'>)`
  * radius: int
  * 生成一个 菱形的 footprint, 指定半径
  * 注意该函数在 radius=1,2 的时候和 disk 相同, 在 3 的时候才会不同
* `skimage.morphology.octagon(m, n, dtype=<class 'numpy.uint8'>)`
  * 生成一个八边形的 footprint, 注意这两个给定参数的意义 
  * m :int. 水平和垂直的4个遍的长度
  * n :int. 斜的四个边的长度, 实际的斜边长度是 n+1, n=0 的时候等于 square
  * 最终会生成一个正方形的 ndarray, 具体的边长会通过 m,n 计算得出

3D template :  
* `skimage.morphology.cube(width, dtype=<class 'numpy.uint8'>)`
* `skimage.morphology.ball(radius, dtype=<class 'numpy.uint8'>)`

## 4.2. 通用 grayscale 变换

Return grayscale morphological transformation of an image.

* `skimage.morphology.dilation(image, footprint=None, out=None, shift_x=False, shift_y=False)`
  * 对每个中心像素, 更新值为局部最大值
* `skimage.morphology.erosion(image, footprint=None, out=None, shift_x=False, shift_y=False)`
  * 对每个中心像素, 更新值为局部最小值  


# 5. feature

包含了多种特征提取函数, 一键调用

## 5.1. hog

输入:
* image : (M, N[, C]) ndarray  注意维度, channel 在最后, 类型是 ndarray
* 除了image 其他参数都有默认值

HOG参数:
* orientations          : int
* pixels_per_cell       : 2-tuple (int, int)
* cells_per_block       : 2-tuple (int, int)
* block_norm            : Block normalization method, 默认 "L2-Hys"
  * L1          Normalization using L1-norm.
  * L1-sqrt     Normalization using L1-norm, followed by square root.
  * L2          Normalization using L2-norm.
  * L2-Hys      Normalization using L2-norm, followed by limiting the maximum values to 0.2 (Hys stands for hysteresis) and renormalization using L2-norm.

设置项:
* feature_vector    : 默认True, 会将特征以一维返回
* visualize         : 默认false, 设置True时会一并返回可视化的 HOG

返回值：
* 注意返回值是 double, 对于只接受 float32 的网络来说需要加上类型转换
* `feature.hog(...).astype(numpy.float32)`

```py
skimage.feature.hog(
    image, 
    orientations=9, 
    pixels_per_cell=(8, 8), 
    cells_per_block=(3, 3), 
    block_norm='L2-Hys', 
    visualize=False, 
    transform_sqrt=False, 
    feature_vector=True, 
    multichannel=None
    )
""" 
return-
out         : (n_blocks_row, n_blocks_col, n_cells_row, n_cells_col, n_orient) ndarray
                HOG descriptor for the image. If feature_vector is True, a 1D (flattened) array is returned.
hog_image   : (M, N) ndarray, optional
                A visualisation of the HOG image. Only provided if visualize is True.
"""
```

# 6. util

提供了面对图像的各种基础工具化的操作方法  

## 6.1. random_noise

`skimage.util.random_noise(image, mode='gaussian', seed=None, clip=True, **kwargs)`  
* image : 一个 ndarry , 将会被转换成 float 类型
* mode  : str, 指定噪声的模式
  * gaussian : 高斯分布的加法噪声
  * localvar : 同样高斯分布, 但是对每个像素都有单独的 variance, 需要传入 local_vars
  * poisson  : 泊松分布
  * salt     : 盐噪声, 白色像素点
  * pepper   : 胡椒噪声, 黑色像素点 
* var   : float , 随机分布的标准差, 被用于 高斯分布和 斑点噪声
  * variance = sigma**2
* local_vars : Array of positive floats. same shape as image.
* mean  : float , 噪声的均值  


* clip  : bool , 用于防止添加噪声后的数据范围越界
* seed  : int, 随机化的种子, 用于数据再现