# 1. scikit-image

A collection of algorithms for image processing.

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

## 2.1. imread 

读一个图像, 特点是读取后的类型是 `ndarray`  

```py
skimage.io.imread(
  fname, 
  as_gray=False, 
  plugin=None, 
  **plugin_args)
""" 

"""
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
