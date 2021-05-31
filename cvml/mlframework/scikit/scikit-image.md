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

# 4. feature

包含了多种特征提取函数, 一键调用

## 4.1. hog

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
